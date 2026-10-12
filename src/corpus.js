/**
 * corpus.js — 内置训练语料（原创英文科普散文，无版权风险，零外部依赖）
 *
 * 选择理由：模型需要展示「学到真实的英语句法 + 领域词汇」，而不是背诵重复模板。
 * 语料主题刻意覆盖 AI 与数学的核心概念，使生成样本的语义可被人类直接判读。
 *
 * @author 晨星
 */

export const CORPUS = `Artificial intelligence is the study of how machines can learn, reason and act.
At its heart stands a single idea: a function with many parameters can be shaped by data until it behaves the way we want.
The shaping process is called optimization, and the tool that performs it is called gradient descent.

A neural network is nothing more than a long chain of linear maps separated by nonlinear gates.
Each linear map is a matrix. Each gate is a small function such as the rectified linear unit or the sigmoid.
Stack enough of them together and the chain becomes a universal approximator, able to represent any continuous function on a compact domain.
That is the content of the universal approximation theorem, proved independently by Hornik, Stinchcombe and White, and by Cybenko.

The success of deep learning is surprising because the loss surface of a neural network is not convex.
Gradient descent has no theoretical guarantee of finding a global minimum in such a landscape.
Yet in practice it finds solutions that generalize. This gap between theory and practice remains one of the central open problems of the field.
Recent work suggests that overparameterization changes the geometry of the problem: when a model has far more parameters than data points, most local minima become nearly equivalent in quality.

Attention is the mechanism that made modern language models possible.
The idea is simple. Given a query vector and a set of key vectors, compute a similarity score between the query and each key, normalize the scores with a softmax, and use the resulting weights to average a set of value vectors.
This operation is differentiable, parallelizable, and expressive. It lets every position in a sequence look directly at every other position in a single layer, without the sequential bottleneck of recurrent networks.

The transformer architecture combines attention with feedforward layers, residual connections and normalization.
Residual connections were introduced by He, Zhang, Ren and Sun for image recognition. They give the gradient a short path back to the input, which makes very deep stacks trainable.
Normalization stabilizes the scale of activations. The original transformer used layer normalization. Most modern models use root mean square normalization, which is cheaper and works just as well.

Position must be encoded, because attention by itself is permutation invariant. It sees a bag of words, not a sequence.
Two families of encoding exist. Absolute encodings add a vector that depends on the index. Relative encodings modify the attention score so that it depends on the distance between two positions.
Rotary position embedding belongs to the second family. It rotates the query and the key by an angle proportional to their position. The dot product of a rotated query and a rotated key then depends only on the distance between them.
This property is elegant and useful. It means a model trained on short sequences can often be extended to longer ones.

Efficiency matters as much as accuracy. The attention matrix grows as the square of the sequence length, so both memory and time become expensive for long documents.
The FlashAttention algorithm avoids writing that matrix to memory at all. It processes the score matrix in blocks, keeping a running maximum and a running sum, and rescales the running output whenever the maximum changes.
The result is numerically identical to the naive computation but requires far less memory traffic. Since memory bandwidth, not arithmetic, is the real bottleneck on modern accelerators, the speedup is substantial.

Grouped query attention reduces the size of the cache used during generation.
In standard multi head attention every query head has its own key head and value head. In grouped query attention several query heads share one key head and one value head.
Quality drops only slightly, while the memory needed to store past keys and values shrinks by the ratio of heads to groups. For a model that must serve many users at once, this trade is almost always worth making.

The optimizer decides how parameters move.
Adam and its variant AdamW are the default choice. They keep a running estimate of the first moment and the second moment of each gradient, then divide one by the square root of the other. This makes the step size roughly independent of the gradient magnitude.
A newer method called Muon takes a different view. It treats a weight matrix as a matrix, not as a bag of numbers, and replaces the update with the nearest semi orthogonal matrix.
Computing that matrix exactly would require a singular value decomposition, which is expensive. Instead the method runs a few steps of a Newton Schulz iteration, which converges to the same answer with nothing but matrix multiplications.
Empirically the method trains faster than Adam on the same budget. It is now used in production models.

Tokenization is the quiet engineering problem behind every language model.
Text arrives as a stream of characters. The model needs integers. The standard solution is byte pair encoding, which starts from single bytes and repeatedly merges the most frequent adjacent pair into a new symbol.
After a few hundred merges, common words become single tokens while rare words decompose into recognizable pieces. The vocabulary stays small enough to be practical and large enough to be expressive.
A well designed tokenizer must be reversible. Encoding and then decoding any string must return the original string exactly. This property is easy to state and easy to break, so it is worth testing explicitly.

Evaluation is harder than it looks.
Perplexity is the exponential of the average cross entropy per token. It measures how surprised the model is by held out data. Lower is better, and the number can be compared across tokenizers only with care.
Perplexity correlates with quality but does not capture it. A model can assign high probability to fluent, empty sentences. Human judgment, targeted benchmarks, and careful error analysis remain necessary.

Reproducibility is a discipline, not an accident.
Every random number in a good codebase comes from a seeded generator. Every experiment records its seed, its configuration, and its commit hash. Every claim is backed by a test that another person can run.
A result that cannot be reproduced is not a result. It is an anecdote.

The most reliable way to trust a piece of numerical code is to check invariants.
An invariant is a property that must hold no matter what the inputs are. The softmax of a vector does not change when a constant is added to every entry. A rotation preserves the length of a vector. A probability distribution sums to one.
When two independent implementations of the same function disagree, one of them is wrong. Writing both a naive version and a fast version, then comparing them, catches errors that no amount of reading would find.
This is why good numerical libraries ship with tests that compare analytic gradients against finite differences.

What comes next is uncertain, and that is part of the appeal.
Scaling laws describe how loss falls as parameters, data and compute grow, but they do not say where the ceiling is.
Some researchers argue that better data and better objectives matter more than raw size. Others argue that new architectures will replace attention entirely.
Both may be right. The history of the field suggests that the most durable ideas are the simplest ones: compose, normalize, and optimize.

Mathematics is the language in which all of this is written.
Linear algebra supplies vectors, matrices, eigenvalues and decompositions. Probability supplies distributions, expectations and the chain rule of entropy.
Optimization supplies convexity, duality and the theory of convergence.
Numerical analysis supplies the caution: floating point arithmetic is not exact, cancellation destroys precision, and a stable formulation is often more valuable than a clever one.
A practitioner who knows only the architecture and not the mathematics will eventually be defeated by a bug that only the mathematics can explain.

So the discipline is a loop. State a hypothesis, build a model, measure the result, explain the measurement, and repeat.
The loop is slow. It is also the only thing that has ever worked.
`;

/** 语料统计（供 README / demo 展示） */
export function corpusStats(text = CORPUS) {
  const chars = text.length;
  const words = text.split(/\s+/).filter(Boolean).length;
  const lines = text.split('\n').filter((s) => s.trim()).length;
  return { chars, words, lines };
}
