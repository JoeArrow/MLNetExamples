using Microsoft.ML.Runtime.Api;

namespace SimpleRegression
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}
