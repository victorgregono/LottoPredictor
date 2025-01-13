
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

// Classe que representa uma entrada de dados da Mega Sena
public class MegaSenaEntry
{
    [LoadColumn(2)] public float Bola1;
    [LoadColumn(3)] public float Bola2;
    [LoadColumn(4)] public float Bola3;
    [LoadColumn(5)] public float Bola4;
    [LoadColumn(6)] public float Bola5;
    [LoadColumn(7)] public float Bola6;
}

// Classe que representa a previsão dos números da Mega Sena
public class MegaSenaPrediction
{
    [ColumnName("Score")]
    public required float[] PredictedNumbers { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        // Caminho para o arquivo CSV com os dados
       
        string diretorioProjeto = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
        string dataPath = Path.Combine(diretorioProjeto, "Data", "mega_sena.csv");


        MLContext mlContext = new();

        // Carregar dados com 80% para treino e 20% para teste
        IDataView dataView = mlContext.Data.LoadFromTextFile<MegaSenaEntry>(
            path: dataPath, hasHeader: true, separatorChar: ',');
        var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        // Criar pipeline usando Random Forest
        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(MegaSenaEntry.Bola1), nameof(MegaSenaEntry.Bola2),
            nameof(MegaSenaEntry.Bola3), nameof(MegaSenaEntry.Bola4), nameof(MegaSenaEntry.Bola5), nameof(MegaSenaEntry.Bola6))
            .Append(mlContext.Transforms.CopyColumns("Label", nameof(MegaSenaEntry.Bola1))) // Usar Bola1 como rótulo
            .Append(mlContext.Regression.Trainers.FastTree());

        // Treinar o modelo
        var model = pipeline.Fit(trainTestSplit.TrainSet);

        // Avaliar o modelo
        var metrics = mlContext.Regression.Evaluate(model.Transform(trainTestSplit.TestSet));
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");

        // Prever os números mais prováveis
        var allData = mlContext.Data.CreateEnumerable<MegaSenaEntry>(dataView, reuseRowObject: false).ToList();
        var frequency = new Dictionary<int, int>();

        // Contar frequência dos números sorteados
        foreach (var entry in allData.SelectMany(e => new[] { e.Bola1, e.Bola2, e.Bola3, e.Bola4, e.Bola5, e.Bola6 }))
        {
            if (frequency.ContainsKey((int)entry))
                frequency[(int)entry]++;
            else
                frequency[(int)entry] = 1;
        }

        // Ordenar por frequência e exibir os números mais prováveis
        var mostFrequentNumbers = frequency
            .OrderByDescending(f => f.Value)
            .Take(6)
            .Select(f => f.Key);
        Console.WriteLine("Números com maior probabilidade baseados em frequência histórica:");
        // Alterar a cor do texto para vermelho
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine(string.Join(", ", mostFrequentNumbers));
        // Restaurar a cor original do texto
        Console.ResetColor();
    }
}

