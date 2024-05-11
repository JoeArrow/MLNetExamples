using System;
using System.Text;
using System.Collections.Generic;

using Microsoft.ML;

namespace Tokenization
{
    class Program
    {
        public static readonly string cr = Environment.NewLine;

        static void Main(string[] args)
        {
            var context = new MLContext();
            var emptyData = new List<SentimentData>();
            var data = context.Data.LoadFromEnumerable(emptyData);

            var tokenization = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', '.', ',' });

            var tokenModel = tokenization.Fit(data);

            var engine = context.Model.CreatePredictionEngine<SentimentData, SentimentTokens>(tokenModel);

            var tokens = engine.Predict(new SentimentData { Text = "This is a test sentence, and it is a long one." });

            Console.WriteLine($"Output 1...{cr}");
            PrintTokens(tokens);

            var charTokenization = context.Transforms.Text.TokenizeIntoCharactersAsKeys("Tokens", "Text", useMarkerCharacters: false)
                .Append(context.Transforms.Conversion.MapKeyToValue("Tokens"));

            var charTokenModel = charTokenization.Fit(data);

            var charEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentTokens>(charTokenModel);

            var charTokens = charEngine.Predict(new SentimentData { Text = "This is a test sentence, and it is a long one." });

            Console.WriteLine($"Output 2...{cr}");
            PrintTokens(charTokens);
            Console.WriteLine($"{cr}Done...{cr}Scroll Up.{cr}");
        }

        // ------------------------------------------------

        private static void PrintTokens(SentimentTokens tokens)
        {
            var sb = new StringBuilder();

            foreach (var token in tokens.Tokens)
            {
                sb.AppendLine(token);
            }

            Console.WriteLine(sb.ToString());
        }
    }
}
