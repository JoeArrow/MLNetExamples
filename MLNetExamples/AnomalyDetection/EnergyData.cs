﻿
using Microsoft.ML.Data;
using System;

namespace AnomalyDetection
{
    public class EnergyData
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Energy { get; set; }
    }
}
