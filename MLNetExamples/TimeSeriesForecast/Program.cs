using System;
using System.Dynamic;

using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace TimeSeriesForecast
{
    class Program
    {
        public static readonly string cr = Environment.NewLine;

        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<RunData>("./RunTimes.csv",
                hasHeader: true, separatorChar: ',');

            var pipeline = context.Forecasting.ForecastBySsa(
                 nameof(RunForecast.RunTime),                   // Output column name for forecasted values
                 nameof(RunData.RunTime),                       // Input column name for the time series data (job run time)
                 windowSize: 30,                                 // Size of the sliding window for SSA analysis
                 seriesLength: 65,                             // Expected length of the time series pattern
                 trainSize: 65,                                 // Size of the training dataset
                 horizon: 1); 

            var model = pipeline.Fit(data);

            var forecastingEngine = model.CreateTimeSeriesEngine<RunData, RunForecast>(context);

            //var input = new RunData() { Job = "WYN_CYCLE_START_C", Month = 5, Day = 11, Year = 24 };
            var input = new RunData() { Job = "WYN_ACTIVITYPROCESSORBATCHMANAGERPRIORITY3_DEP_C", Month = 5, Day = 11, Year = 24 };

            var forecasts = forecastingEngine.Predict(input);

            Console.WriteLine($"Job: {input.Job}{cr}");

            foreach (var forecast in forecasts.RunTime)
            {
                Console.WriteLine($"Forecast: {Math.Abs(forecast)}");
            }
        }
    }
}
