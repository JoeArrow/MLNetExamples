using System;
using System.IO;
using System.Linq;
using System.Drawing;

using Microsoft.ML;
using Microsoft.ML.Vision;

namespace DeepNeuralNetworkUpdate
{
    class Program
    {
        static void Main(string[] args)
        {
            var cr = Environment.NewLine;
            var modelPath = "./dnn_model.mdl";
            ITransformer model;

            var context = new MLContext();

            if(File.Exists(modelPath))
            {
                Console.WriteLine($"Loading the model...{cr}{cr}");
                model = context.Model.Load(modelPath, out DataViewSchema inputSchema);
            }
            else
            {
                var imagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "images");
                var files = Directory.GetFiles(imagesFolder, "*", SearchOption.AllDirectories);

                var images = files.Select(file => new ImageData
                {
                    ImagePath = file,
                    Label = Directory.GetParent(file).Name
                });

                var imageData = context.Data.LoadFromEnumerable(images);
                var imageDataShuffled = context.Data.ShuffleRows(imageData);
                var dataSets = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);

                var validationData =
                    context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                    .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"))
                    .Fit(dataSets.TestSet)
                    .Transform(dataSets.TestSet);

                var imagesPipeline =
                    context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                    .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"));

                var imageDataModel = imagesPipeline.Fit(dataSets.TrainSet);
                var imageDataView = imageDataModel.Transform(dataSets.TrainSet);

                var options = new ImageClassificationTrainer.Options()
                {
                    Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                    Epoch = 100,
                    BatchSize = 20,
                    LearningRate = 0.01f,
                    LabelColumnName = "LabelKey",
                    FeatureColumnName = "Image",
                    ValidationSet = validationData
                };

                var pipeline = context.MulticlassClassification.Trainers.ImageClassification(options)
                    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                Console.WriteLine($"Training Model...{cr}");

                model = pipeline.Fit(imageDataView);

                context.Model.Save(model, imageData.Schema, modelPath);
                Console.WriteLine($"{cr}Model Saved...{cr}{modelPath}{cr}");
            }

            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test");
            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

            var testImages = testFiles.Select(file => new ImageModelInput
            {
                ImagePath = file,
                Image = ImageToByteArray(Image.FromFile(file))
            });

            // ---------------

            var predictionEngine = context.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(model);

            var line = new string('=', 50);
            Console.WriteLine($"{cr}{line}{cr}");

            foreach(var input in testImages)
            {
                var myPred = predictionEngine.Predict(input);
                Console.WriteLine($"Image: {Path.GetFileName(myPred.ImagePath)} - Predicted Label: {myPred.PredictedLabel}");
            }

            Console.WriteLine($"{cr}{line}{cr}");
        }

        // ------------------------------------------------

        static byte[] ImageToByteArray(Image image)
        {
            using(MemoryStream stream = new MemoryStream())
            {
                image.Save(stream, System.Drawing.Imaging.ImageFormat.Jpeg);
                return stream.ToArray();
            }
        }
    }
}
