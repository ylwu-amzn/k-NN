/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.SetOnce;
import org.opensearch.action.ActionListener;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.ml.client.MachineLearningClient;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelResultFilter;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

@Log4j2
@Getter @Setter
public class NlpQueryBuilder extends AbstractQueryBuilder<NlpQueryBuilder> {

    public static final String NAME = "neural";
    public static final String DOC_FIELD = "doc";
    public static final String MODEL_ID_FIELD = "model_id";
    public static final String KNN_VECTOR_FIELD = "knn_vector_field";
    public static final String K_FIELD = "k";

    public static MachineLearningClient mlClient;
    private float[] vector;



    private String doc;
    private String modelId;
    private String knnVectorField;
    private int k;
    public Supplier<float[]> supplier;

    NlpQueryBuilder() {}

    public static NlpQueryBuilder fromXContent(XContentParser parser) throws IOException {

        String doc = null;
        String modelId = null;
        String knnVectorField = null;
        int k = 10;
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();
            switch (fieldName) {
                case DOC_FIELD:
                    doc = parser.text();
                    break;
                case MODEL_ID_FIELD:
                    modelId = parser.text();
                    break;
                case KNN_VECTOR_FIELD:
                    knnVectorField = parser.text();
                    break;
                case K_FIELD:
                    k = parser.intValue();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        if (doc == null) {
            throw new IllegalArgumentException("no doc");
        }
        if (modelId == null) {
            throw new IllegalArgumentException("no model id");
        }
        if (knnVectorField == null) {
            throw new IllegalArgumentException("no knn vector field");
        }

        NlpQueryBuilder nlpQueryBuilder = new NlpQueryBuilder();
        nlpQueryBuilder.setDoc(doc);
        nlpQueryBuilder.setModelId(modelId);
        nlpQueryBuilder.setKnnVectorField(knnVectorField);
        nlpQueryBuilder.setK(k);
        return nlpQueryBuilder;
    }

    public NlpQueryBuilder(StreamInput in) throws IOException {
        super(in);
        this.doc = in.readString();
        this.modelId = in.readString();
        this.knnVectorField = in.readString();
        this.k = in.readInt();
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(doc);
        out.writeString(modelId);
        out.writeString(knnVectorField);
        out.writeInt(k);
    }

    public NlpQueryBuilder(Supplier<float[]> supplier) {
        this.vector = supplier.get();
    }

    public NlpQueryBuilder newNlpQueryBuilder(Supplier<float[]> supplier) {
        return new NlpQueryBuilder(supplier);
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext queryRewriteContext) throws IOException {
        if (supplier != null) {
            return supplier.get() == null ? this : new KNNQueryBuilder(knnVectorField, supplier.get(), k);
        }
        if (this.vector == null) {
            SetOnce<float[]> supplier = new SetOnce<>();
            queryRewriteContext.registerAsyncAction((client, listener) -> {
                // Model may output multiple results. "targetResponse" is to filter which results you want to return,
                // For example, model output "input_ids" and "sentence_embedding". We only need "sentence_embedding".
                // Then we can just set "sentence_embedding" in targetResponse.
                // If targetResponse is null, will return all outputs.
                List<String> targetResponse = Arrays.asList("sentence_embedding");
                // Some model's output doesn't have a key or name. User can specify the result position to get target output.
                // For example the model has three outputs and we may only need the second one, then just add 1 to
                // targetResponsePosition.
                // If targetResponsePositions is null, will return all output.
                List<Integer> targetResponsePositions = null;
                ModelResultFilter filter = new ModelResultFilter(false, true, targetResponse, targetResponsePositions);
                MLInputDataset inputDataset = new TextDocsInputDataSet(Arrays.asList(doc), filter);
                MLInput mlInput = new MLInput(FunctionName.TEXT_EMBEDDING, null, inputDataset);
                mlClient.predict(this.modelId, mlInput, ActionListener.wrap(mlOutput -> {
                    ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlOutput;
                    ModelTensors modelTensors = modelTensorOutput.getMlModelOutputs().get(0);
                    float[] vector = null;
                    for (ModelTensor out : modelTensors.getMlModelTensors()) {
                        if ("sentence_embedding".equals(out.getName())) {
                            vector = new float[out.getData().length];
                            for (int i = 0 ; i<out.getData().length;i++) {
                                vector[i] = out.getData()[i].floatValue();
                            }
                            this.setVector(vector);
                        }
                    }
                    if (vector == null) {
                        throw new IllegalArgumentException("No vector generated");
                    }
                    log.debug("Generated vector for \"{}\" : {} ", doc, Arrays.toString(vector));
                    supplier.set(vector);
                    listener.onResponse(null);
                }, e -> {
                    log.error("Failed to call ml-commons", e);
                    listener.onFailure(e);
                }));
            });
            NlpQueryBuilder nlpQueryBuilder = new NlpQueryBuilder();
            nlpQueryBuilder.setSupplier(supplier::get);
            nlpQueryBuilder.setDoc(doc);
            nlpQueryBuilder.setModelId(modelId);
            nlpQueryBuilder.setKnnVectorField(knnVectorField);
            nlpQueryBuilder.setK(k);
            return nlpQueryBuilder;
        }
        return this;
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.field(DOC_FIELD, doc);
        builder.field(MODEL_ID_FIELD, modelId);
        builder.field(KNN_VECTOR_FIELD, knnVectorField);
        builder.field(K_FIELD, k);
        builder.endObject();
    }

    @Override
    protected Query doToQuery(QueryShardContext context) throws IOException {
        String indexName = context.index().getName();
        return KNNQueryFactory.create(KNNEngine.DEFAULT, indexName, this.knnVectorField, this.vector, this.k);
    }

    @Override
    protected boolean doEquals(NlpQueryBuilder nlpQueryBuilder) {
        return true;
    }

    @Override
    protected int doHashCode() {
        return 0;
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
