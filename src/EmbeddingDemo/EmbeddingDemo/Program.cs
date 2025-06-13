using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

class Program
{
    static void Main()
    {
        var vocab = new Dictionary<string, long>
        {
            { "the", 0 },
            { "cat", 1 },
            { "sat", 2 },
            { "on", 3 },
            { "mat", 4 }
        };

        string[] sentence = new[] { "the", "cat", "sat" };
        long[] tokenIds = sentence.Select(word => vocab[word]).ToArray();

        torch.random.manual_seed(42);
        var device = torch.CPU;

        var embedding = nn.Embedding(vocab.Count, 8).to(device);
        var input = torch.tensor(tokenIds, dtype: ScalarType.Int64).to(device);
        var vectors = embedding.forward(input);

        Console.WriteLine("Token IDs: " + string.Join(", ", tokenIds));
        Console.WriteLine("\nEmbeddings:");
        for (int i = 0; i < vectors.shape[0]; i++)
        {
            var vector = vectors[i];
            Console.Write($"{sentence[i],-5} → ");
            Console.WriteLine(string.Join(", ", vector.data<float>().ToArray()));
        }
    }
}
