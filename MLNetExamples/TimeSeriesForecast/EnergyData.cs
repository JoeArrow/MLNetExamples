using System;

using Microsoft.ML.Data;

namespace TimeSeriesForecast
{
    public class EnergyData
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Energy { get; set; }
    }
}
