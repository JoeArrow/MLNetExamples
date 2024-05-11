using System;

using Microsoft.ML.Data;

namespace TimeSeriesForecast
{
    // ----------------------------------------------------
    /// <summary>
    ///     RunData Description
    /// </summary>

    public class RunData
    {
        [LoadColumn(0)]
        public string Job { get; set; }

        [LoadColumn(1)]
        public Single RunTime { get; set; }

        [LoadColumn(2)]
        public Single Day { get; set; }

        [LoadColumn(3)]
        public Single Month { get; set; }

        [LoadColumn(4)]
        public Single Year { get; set; }
    }
}
