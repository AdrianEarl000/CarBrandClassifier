using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CarBrandClassifier
{

    public class ModelInput
    {
        [LoadColumn(0)]
        public byte[] ImageSource { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }


    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }


    }
}
