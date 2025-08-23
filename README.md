# AI-deeplearning-noob-
This is a simple AI deep learning program. It's a really simple program, and it serves as the framework for the deep learning portion of the AI ​​project I'm currently working on.
AI name: M
code (csharp):
---


using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

class ByteTokenizer
{
    public Tensor Encode(string s) => tensor(Encoding.UTF8.GetBytes(s).Select(b => (long)b).ToArray(), dtype: ScalarType.Int64);
    public string Decode(Tensor t)
    {
        var bytes = t.to(ScalarType.Int64).data<long>().ToArray().Select(i => (byte)(i & 0xFF)).ToArray();
        return Encoding.UTF8.GetString(bytes);
    }
}

class ByteLMDataset
{
    readonly long[] data;
    readonly int ctx;
    readonly Random rng = new Random(1234);
    public ByteLMDataset(string path, int ctxLen)
    {
        if (!File.Exists(path)) File.WriteAllText(path, "User: Hello\nM: Hi there. How can I help you?\n\nUser: Introduce yourself\nM: I am a small language model built with C# and TorchSharp.\n");
        var bytes = File.ReadAllBytes(path);
        data = bytes.Select(b => (long)b).ToArray();
        ctx = ctxLen;
    }
    public (Tensor x, Tensor y) GetBatch(int batch)
    {
        if (data.Length <= ctx + 1) throw new Exception("Corpus is too short.");
        var xs = new List<Tensor>();
        var ys = new List<Tensor>();
        for (int i = 0; i < batch; i++)
        {
            int start = rng.Next(0, data.Length - ctx - 2);
            var x = tensor(data.Skip(start).Take(ctx).ToArray(), dtype: ScalarType.Int64);
            var y = tensor(data.Skip(start + 1).Take(ctx).ToArray(), dtype: ScalarType.Int64);
            xs.Add(x.unsqueeze(0));
            ys.Add(y.unsqueeze(0));
        }
        return (cat(xs.ToArray(), 0), cat(ys.ToArray(), 0));
    }
}

class ByteLM : nn.Module
{
    readonly int vocab;
    readonly int ctx;
    readonly Embedding tokEmb;
    readonly Embedding posEmb;
    readonly TransformerEncoder encoder;
    readonly LayerNorm lnF;
    readonly Linear head;
    readonly Dropout drop;
    public ByteLM(int vocabSize, int ctxLen, int dModel, int nHead, int nLayer, double dropout) : base("bytelm")
    {
        vocab = vocabSize;
        ctx = ctxLen;
        tokEmb = nn.Embedding(vocab, dModel);
        posEmb = nn.Embedding(ctx, dModel);
        var layers = new TransformerEncoderLayer(dModel, nHead, 4 * dModel, dropout, activation: TorchSharp.Modules.Activations.GELU);
        encoder = nn.TransformerEncoder(layers, nLayer);
        lnF = nn.LayerNorm(new long[] { dModel });
        head = nn.Linear(dModel, vocab, hasBias: false);
        drop = nn.Dropout(dropout);
        RegisterComponents();
    }
    Tensor CausalMask(int T, Device device)
    {
        var m = ones(new long[] { T, T }, device: device).triu(1);
        m = m.masked_fill(m.eq(1), float32.MinValue);
        return m;
    }
    public (Tensor logits, Tensor loss) Forward(Tensor idx, Tensor targets = null)
    {
        var device = idx.device;
        var B = idx.shape[0];
        var T = idx.shape[1];
        var pos = arange(0, T, device: device, dtype: ScalarType.Int64).unsqueeze(0).expand(B, T);
        var x = tokEmb.forward(idx) + posEmb.forward(pos);
        x = drop.forward(x);
        var mask = CausalMask(T, device);
        x = encoder.forward(x.transpose(0, 1), mask).transpose(0, 1);
        x = lnF.forward(x);
        var logits = head.forward(x);
        Tensor loss = null;
        if (targets is not null)
        {
            loss = nn.functional.cross_entropy(logits.view(-1, vocab), targets.view(-1));
        }
        return (logits, loss);
    }
    public Tensor Generate(Tensor idx, int maxNewTokens, double temperature = 0.9, int topK = 50)
    {
        using var _ = no_grad();
        var device = idx.device;
        for (int t = 0; t < maxNewTokens; t++)
        {
            var idxCond = idx.slice(1, Math.Max(0, (int)idx.shape[1] - ctx));
            var (logits, _) = Forward(idxCond, null);
            var last = logits[TensorIndex.Ellipsis, -1, TensorIndex.Ellipsis] / Math.Max(1e-6, temperature);
            if (topK > 0 && topK < vocab)
            {
                var (v, ix) = last.topk(topK);
                var mask = last < v[TensorIndex.Ellipsis, TensorIndex.Slice(new TensorIndex[] { TensorIndex.None, TensorIndex.None }, -1)];
                last = last.masked_fill(mask, float32.MinValue);
            }
            var probs = nn.functional.softmax(last, -1);
            var nextId = multinomial(probs, 1);
            idx = cat(new Tensor[] { idx, nextId }, 1);
        }
        return idx;
    }
}

class Program
{
    static void Main(string[] args)
    {
        torch.random.manual_seed(1234);
        var device = cuda.is_available() ? CUDA : CPU;
        int vocab = 256;
        int ctx = 256;
        int dModel = 512;
        int nHead = 8;
        int nLayer = 6;
        double dropout = cuda.is_available() ? 0.1 : 0.0;
        int batch = 16;
        int steps = 500;
        double lr = 3e-4;
        string corpusPath = "dialog.txt";
        if (!File.Exists(corpusPath))
        {
            File.WriteAllText(corpusPath, "User: Hello\nM: Hi there. How can I help you?\n\nUser: Who are you\nM: I am a small language model built with C# and TorchSharp.\n");
        }
        var dataset = new ByteLMDataset(corpusPath, ctx);
        var model = new ByteLM(vocab, ctx, dModel, nHead, nLayer, dropout).to(device);
        var optim = torch.optim.AdamW(model.parameters(), lr: lr, betas: (0.9, 0.95), weight_decay: 0.01);
        Console.WriteLine("ready");
        var tok = new ByteTokenizer();
        for (int i = 0; i < steps; i++)
        {
            model.train();
            var (x, y) = dataset.GetBatch(batch);
            x = x.to(device);
            y = y.to(device);
            var (_, loss) = model.Forward(x, y);
            optim.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0);
            optim.step();
            if ((i + 1) % 50 == 0) Console.WriteLine($"step {i + 1}/{steps} loss {loss.ToSingle():0.4f}");
        }
        while (true)
        {
            Console.Write("input > ");
            var line = Console.ReadLine();
            if (line == null || line.Trim().ToLower() == "quit") break;
            var prompt = $"User: {line}\nM:";
            var idx = tok.Encode(prompt).unsqueeze(0).to(device);
            var outIds = model.Generate(idx, 256, temperature: 0.9, topK: 50).cpu()[0];
            var gen = tok.Decode(outIds.slice(0, (long)idx.shape[1]).ToInt64Array().Length == outIds.shape[0] ? tensor(Array.Empty<long>()) : outIds.slice(0, (long)idx.shape[1]));
            var full = tok.Decode(outIds);
            var reply = full.Substring(tok.Decode(idx[0]).Length);
            var cut = reply.Split("\nUser:")[0].Trim();
            Console.WriteLine("M > " + (string.IsNullOrWhiteSpace(cut) ? "(...)" : cut));
            Console.Write("teach > ");
            var teach = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(teach))
            {
                File.AppendAllText(corpusPath, $"\nUser: {line}\nM: {teach}\n");
                dataset = new ByteLMDataset(corpusPath, ctx);
                for (int i = 0; i < 200; i++)
                {
                    model.train();
                    var (x, y) = dataset.GetBatch(batch);
                    x = x.to(device);
                    y = y.to(device);
                    var (_, loss) = model.Forward(x, y);
                    optim.zero_grad();
                    loss.backward();
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0);
                    optim.step();
                }
            }
        }
    }
}
