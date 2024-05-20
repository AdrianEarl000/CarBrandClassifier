using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CarBrandClassifier;
using Microsoft.ML;


namespace CarBrandClassifier
{
    public class MLLoader
    {
        private string modelFilePath;
        private MLContext mlContext;
        private ITransformer mlModel;

        public MLLoader(string modelFilePath)
        {
            this.modelFilePath = modelFilePath;
            mlContext = new MLContext();


            using (var stream = new System.IO.FileStream(modelFilePath, System.IO.FileMode.Open))
            {
                mlModel = mlContext.Model.Load(stream, out var schema);
            }
        }

        public ModelOutput Predict(ModelInput input)
        {

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
            var result = predictionEngine.Predict(input);
            return result;
        }
    }
}
