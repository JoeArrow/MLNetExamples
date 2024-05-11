using System;
using Microsoft.ML;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime.Data;

namespace SimpleRegression
{
    class Program
    {
        public static readonly string cr = Environment.NewLine;

        static void Main(string[] args)
        {
            var yoe = 8;
            var context = new MLContext();

            var reader = TextLoader.CreateReader(context, ctx => (
                YearsExperience: ctx.LoadFloat(0),
                Salary: ctx.LoadFloat(1)
            ), hasHeader: true, separator: ',');

            var data = reader.Read(new MultiFileSource("SalaryData.csv"));

            var pipeline = reader.MakeNewEstimator()
                .Append(r => (r.Salary, Prediction: context.Regression.Trainers.Sdca(label: r.Salary, features: r.YearsExperience.AsVector())));

            var model = pipeline.Fit(data).AsDynamic;

            var predictionFunc = model.MakePredictionFunction<SalaryData, SalaryPrediction>(context);

            var prediction = predictionFunc.Predict(new SalaryData { YearsExperience = yoe });

            Console.WriteLine($"Years of Experience: {yoe}{cr}Predicted salary - {string.Format("{0:C}", prediction.PredictedSalary)}{cr}");
        }
    }
}
