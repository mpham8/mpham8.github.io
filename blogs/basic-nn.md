# Math and Implementation of Neural Network from Scratch

**Michael Pham**  
*April 11, 2026*

<figure>
  <img src="blogs/basic-nn-figures/model.png" alt="784→64→10 two-layer network: input, hidden, and output blocks with weight matrices W1 and W2" />
  <figcaption>Model diagram</figcaption>
</figure>


A basic 2 layer neural network [input (784) → hidden (64) → output (10)] trained with Adam optimizer and mini-batch SGD, with inference implemented in C++.

It's been a while since I've done any machine learning work beyond calling Transformers from HuggingFace. I wanted to brush up on the fundamentals, so I decided the best way was to implement a neural net from scratch.


[I actually did this back in 2022](https://github.com/mpham8/Number-Identifier-Scratch-ANN/blob/main/model.ipynb). At the time I was pretty intimidated by matrix operations — I had just taken Linear Algebra the semester before, so a lot of it went over my head (although, linear algebra really isn't really needed to follow. I think some understanding on matrix multiplication and calculus to the topic of partial derivatives is enough to follow along). At the time I implemented by mostly just copying down the math from [this video](https://www.youtube.com/watch?v=w8yWXqWQYmU).

Since the math can be intimidating, I figured I'd rework it out as an exercise — both for myself and for anyone (or me, 4 years ago) trying to understand what's actually going on under the hood. If you've worked with neural networks or pyTorch but find yourself a little fuzzy on the math, this is written for you. If you don't yet understand neural networks, just watch [this Stanford CS229 lecture explaining neural networks](https://www.youtube.com/watch?v=MfIjxPh6Pys&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=11) and come back to this write-up.

You can view [all the code referenced in this write-up here: https://github.com/mpham8/nn-fundamental/blob/main/nn-torchless/src-cpp/main.cpp](https://github.com/mpham8/nn-fundamental/blob/main/nn-torchless/src-cpp/main.cpp).

## Model

I'll be working with the MNIST dataset, a collection of labeled (zero through nine) digit images, each consisting of 28x28 pixels. When flattened, each image i becomes 784 pixels or in vector form a 784-element vector \(\vec{x}^{(i)}\). As to not get confused by my notation, \((i)\) refers to the ith training example and \([i]\) refers to the ith layer in the neural network. As seen in the model diagram, this serves as the input layer to the neural net. Stacking all m examples horizontally creates a matrix \(X\), which is 784 x m:

<figure>
  <img src="blogs/basic-nn-figures/input-matrix.png" alt="Input Matrix (784 x m)" />
  <figcaption>Input Matrix (784 x m)</figcaption>
</figure>


## Forward Pass

The first part of the code is the forward propagation. The first set of weights, \( w^{[1]} \), maps the 784 input features down to the 64 nodes in the first hidden layer. Since we want \( w^{[1]} X \) to produce a 64 x m matrix (one column per training example), and \( X \) is 784 x m, \( w^{[1]} \) needs to be 64 x 784 - this follows from the standard rule that a x b matrix times b x c matrix outputs a x c matrix. The bias term \( b^{[1]} \) is then added element-wise, getting us our first output \( z^{[1]} \).

$$\underset{64 \times m}{z^{[1]}} = \underset{64 \times 784}{w^{[1]}} \quad\underset{784 \times m}{X} + \underset{64 \times m}{b^{[1]}}$$

This first linear (affine) transformation is represented in the forward method as 

```cpp
Matrix h = Linear(x, w1, b1);
z1 = h;
```

We also store this first output \( z^{[1]} \). This might seem unnecessary for now, but it will become clear why it's needed later, when backpropagation is explained. The `Linear` method is

```cpp
Matrix nn::Linear(const Matrix &x, const Matrix &w, const Matrix &b) {
  const Matrix bBroadcast = b.replicate(1, x.cols());
  return w * x + bBroadcast;
}
```

This seems straight-forward except for the fact that \( b^{[1]} \) is actually a 64 x 1 vector. If there was only one training example, this would be straight-forward as \( w^{[1]} X\) would be 64 x 1 and then you can just element-wise sum a 64x1 vector with another 64x1 vector. However, as we previously worked out, with \( w^{[1]} X\) you get a 64 x m vector, which you can't sum with a 64 x 1 vector. The bias is going to hit all m columns the same way, so we're going to broadcast the bias, i.e. just copy and paste it m times as follows:

<figure>
  <img src="blogs/basic-nn-figures/broadcast.png" alt="broadcast" />
  <figcaption>Broadcasting</figcaption>
</figure>

Then, we take this output \( z^{[1]} \) and pass it through an activation layer, ReLU, to get the activated node \( a^{[1]}\):

$$a^{[1]} = \sigma(z^{[1]})$$

simply in the code (where a1 is also stored for later)

```cpp
h = ReLu(h);
a1 = h;
```

where ReLU is defined as 

$$
\sigma_{ReLU}(x) = \max(0, x)
$$

```cpp
Matrix nn::ReLu(const Matrix &x) { return x.cwiseMax(0.0f); }
```

In other words, for each element in the matrix, negative values are set to zero and positive values remain unchanged. ReLU is just taking an element-wise max comparator to 0 so this shouldn't change the output dimension of \( a^{[1]}\). As such, \( a^{[1]}\) remains 64 x m. From here, we do another linear transformation to \( a^{[1]}\) to get it to our second layer of 10 nodes. 


$$\underset{10 \times m}{z^{[2]}} = \underset{10 \times 64}{w^{[2]}} \quad \underset{64 \times m}{a^{[1]}} + \underset{10 \times m}{b^{[2]}}$$


In the code, $z^{[2]}$ is referred to as logits since $z^{[2]}$ is the set of unnormalized, raw "scores" produced by the last layer of the network. The output layer has 10 nodes: one per class where each node's score represents the unnormalized prediction for that digit.

```cpp
const Matrix logits = Linear(h, w2, b2);
```

Then, another activation function is applied onto $z^{[2]}$, which again doesn't change the dimensions from $z^{[2]}$.

$$\hat{y} = \sigma(z^{[2]})$$

```cpp
probs = softmax(logits);
```

where this time the activation function is softmax.

\[
\sigma_{\text{softmax}}(z_k^{(i)}) = \frac{e^{z_k^{(i)}}}{\sum_{j=1}^K e^{z_j^{(i)}}}
\]

where z_k^{(i)} is the ith training example's kth class logit. K is the number of classes (or output nodes), and softmax is applied column-wise across the logit matrix $z^{[2]$, so each column sums to 1 and forms a valid discrete probability distribution over classes, so the i superscript can also be removed to form a valid equation for softmax. The code for the `softmax` function is:

```cpp
Matrix nn::softmax(const Matrix &x) {
  const auto colmax = x.colwise().maxCoeff();
  const Matrix shifted = x - colmax.replicate(x.rows(), 1);
  const Matrix exps = shifted.array().exp();
  const auto colsum = exps.colwise().sum();
  return exps.array().cwiseQuotient(colsum.replicate(x.rows(), 1).array());
}
```

Putting all of the pieces together, the entire forward pass method is written as:

```cpp
Matrix nn::forward(const Matrix &x) {
  Matrix h = Linear(x, w1, b1);
  z1 = h;
  h = ReLu(h);
  a1 = h;
  const Matrix logits = Linear(h, w2, b2);
  probs = softmax(logits);

  return logits;
}
```

If you're coming from pyTorch, this maps similarly to the standard `forward` method:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
```


## Backpropagation

In order to do gradient descent, we're going to need the gradient of the loss function with respect to each of the parameters we're updating. Let's first start by defining a loss function. Cross-entropy loss is a natural choice for multi-class classification. Cross-entropy loss is easy to work with as we'll see and well-suited to problems with mutually exclusive classes.

$$\mathcal{L}_{CE}^{(i)} = -\sum_{k=1}^{K} y_k^{(i)} \log \hat{y}_k^{(i)}$$

Here, $y_k^{(i)} \in \{0, 1\}$ is the one-hot indicator for example $i$, acting as an "activator": it equals 1 for the true class $k$, keeping that $\log \hat{y}_k^{(i)}$ term, and 0 for all others, cancelling them out. The sum across all K classes therefore collapses to a single term: the log-probability $\log \hat{y}_k^{(i)}$ assigned to the true class k. In other words, $\hat{y}_k^{(i)}$ is the softmax output for class $k$, so we are penalizing the model for assigning low probability to the true class. When the model is confident and correct i.e., $\hat{y}_k^{(i)} \approx 1$ for the true class the loss approaches $-\log(1) = 0$. As the predicted probability falls toward zero, the loss grows without bound, since $-\log(x) \to \infty$ as $x \to 0^+$.

The total loss is simply the average cross-entropy across all $m$ training examples:

$$J(\hat{y}, y) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}_{CE}^{(i)}
             = \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{10} y_k^{(i)} \log \hat{y}_k^{(i)}$$

So now that we have the loss function, we can compute gradients. Backpropogation is essentially going to involve a lot of chain rules starting from the last layer (working backwards hence the name backward propagation). We always begin with the parameters furthest from 
the input (i.e., the last layer's weights and biases), because the intermediate 
quantities computed along the way can be reused when computing gradients for earlier 
layers. The natural starting point is therefore $w^{[2]}$ and $b^{[2]}$, the weights 
and biases of the second (output) layer.  So how do we compute $\frac{\partial J}{\partial w^{[2]}}$? Via chain rule, we can decompose $\frac{\partial J}{\partial w^{[2]}}$ as the product of 2 different partial derivatives. This is totally legal; this isn't the same thing, but intuitively think about how you can expand $\frac{a}{b} = \frac{a}{c} * \frac{c}{b}$ where the c's can cancel out.

<figure>
  <img src="blogs/basic-nn-figures/dJdW2.png" alt="dJdW2" />
  <figcaption>$\frac{\partial J}{\partial w^{[2]}}$</figcaption>
</figure>


You can either remember that always that for the derivative of cross entropy loss function with respect to the input of the final softmax layer simplifies neatly to $\frac{1}{m}(\hat{y} - y)$

$$\frac{\partial J}{\partial z^{[2]}} = \frac{1}{m}(\hat{y} - y)$$ 

or you can [read this derivation](blogs/basic-nn-figures/dJdZ2-derivation.pdf) (I omit the ith superscript for most of it for notation simplicity).

And then $\frac{\partial z^{[2]}}{\partial w^{[2]}} = (a^{[1]})^T$. Think in the scalar case if these were all scalars and $z = aw + b$. $\frac{\partial z}{\partial x} = w$. Because now with the full matrices this is also a linear transformation, the partial derivative is also $a^{[1]}$. But if you notice both $\hat{y}$ and $y$ are 10 x m vectors so $\frac{1}{m}(\hat{y} - y)$ is also a 10 x m vectors. so we need $\frac{\partial z^{[2]}}{\partial w^{[2]}}$ to have m rows in order to be able to do matrix multiplication. Since $a^{[1]}$ is 64 x m, we can simply transpose $a^{[1]}$.

Both $\hat{Y}$ and $Y$ are 10 x m matrices, so their difference $\frac{1}{m}(\hat{Y} - Y)$ is also 10 x m. For the matrix multiplication to work when computing the gradient, we use $a^{[1]}$ transposed, which is m x 64, so the shapes align: the result is 10 x 64, which matches the dimensions of $w^{[2]}$.

<figure>
  <img src="blogs/basic-nn-figures/dJdB2.png" alt="dJdB2" />
  <figcaption>$\frac{\partial J}{\partial b^{[2]}}$</figcaption>
</figure>

A similar process is done for $\frac{\partial J}{\partial b^{[2]}}$, however, a caveat is that if you remember, we broadcasted $b^{[2]}$. If $b^{[2]}$ was just a 10 x 1 vector, then $\frac{\partial J}{\partial b^{[2]}} = 1$ (think back to the scalar analogy where in $z = aw + b$, $\frac{\partial z}{\partial b} = 1$). 

However, due to $b^{[2]}$ being broadcasted to be 10 x m,  $\frac{\partial z}{\partial b} = \mathbf{1}$ where $\mathbf{1}$ is a m x 1 vector of ones. Multiplying a matrix by $\mathbf{1}$ sums across the columns, in other words each row gets added up:

$$\begin{pmatrix} \mid & \mid & & \mid \\ \delta^{(1)} & \delta^{(2)} & \cdots & \delta^{(m)} \\ \mid & \mid & & \mid \end{pmatrix} \begin{pmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{pmatrix} = \sum_{i=1}^{m} \delta^{(i)}$$

This is why the final step also sums across the columns to get $\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})$. Now that we've derived $\frac{\partial J}{\partial w^{[2]}}$ and $\frac{\partial J}{\partial b^{[2]}}$, we can use some of these derivatives we already computed to calculate the next backwards gradients: $\frac{\partial J}{\partial w^{[1]}}$ and $\frac{\partial J}{\partial b^{[1]}}$. $\frac{\partial J}{\partial w^{[1]}}$ can be decomposed via the chain rule as:

<figure>
  <img src="blogs/basic-nn-figures/dJdW1.png" alt="dJdW1" />
  <figcaption>$\frac{\partial J}{\partial w^{[1]}}$</figcaption>
</figure>

We've already calculated $\frac{\partial J}{\partial z^{[2]}}$ in the previous layer. Similar to how $\frac{\partial z^{[2]}}{\partial w^{[2]}} = a^{[1]\top}$, $\frac{\partial z^{[2]}}{\partial a^{[1]}} = w^{[2]}$ which is 10 x 64. $\frac{\partial a^{[1]}}{\partial z^{[1]}} $ is asking us to calculate the derivative of the ReLU activation function wrt to inputs. For ReLU, if $x \geq 0$ then $\sigma(x) = x$, which means that its derivative wrt x is 1 (the derivative of a linear function with slope 1 is 1). If $x < 0$, then $\sigma(x) = 0$, which means that its derivative wrt x is 0 (the derivative of a flat line is 0). Therefore, the derivative of the ReLU function wrt to its inputs is 1 if $x > 0$ otherwise it's 0. A fancy way of writing this is with an indicator $\mathbf{1}[z^{[1]} > 0]$. Since ReLU was done element-wise, since both $a^{[1]}$ and $z^{[1]}$ were 64 x m, so is the derivative. Finally, the $\frac{\partial z^{[1]}}{\partial w^{[1]}}$ is the input matrix itself.

In order for matrix multiplication to work, things need to be rearranged and transposed so that the clumns of each matrix matches the rows of the next matrix, hence the final step. 

$\frac{\partial J}{\partial b^{[1]}}$ follows:

<figure>
  <img src="blogs/basic-nn-figures/dJdB1.png" alt="dJdB1" />
  <figcaption>$\frac{\partial J}{\partial b^{[1]}}$</figcaption>
</figure>

Deriving all the gradients may have been complicated, but the implementation is a lot more straight-forward: 

```cpp
void nn::backward(const Matrix &x, const Eigen::VectorXf &y) {
  const Eigen::Index m = x.cols();
  const int numClasses = static_cast<int>(w2.rows());
  const Matrix Y = oneHotEncoding(y, numClasses);

  const Matrix dJdz2 = (probs - Y) / static_cast<float>(m);
  dJdw2 = dJdz2 * a1.transpose();
  dJdb2 = dJdz2.rowwise().sum();
  const Matrix dJda1 = w2.transpose() * dJdz2;
  const Matrix dJdz1 = dJda1.cwiseProduct((z1.array() > 0.f).cast<float>().matrix());
  dJdw1 = dJdz1 * x.transpose();
  dJdb1 = dJdz1.rowwise().sum();
}
```

The reason why we stored \( a^{[1]} \) and \( z^{[1]} \) during forward propagation earlier is because these matrices are required to compute some gradients during backpropagation.


In pyTorch, the reason why `backward()` is called during training is that this method similarly calculates all of the gradients needed for the parameter update step (in a less tedious, manual way).
```python
loss.backward()
```

## Mini-Batch Stochastic Gradient Descent

### Update Parameters with Adam

Instead of the standard gradient descent update $\theta := \theta - \alpha \nabla_\theta J(\theta)$, we use Adam, which only adds a couple of variables to keep track of. Adam and Adam variants are the de facto default in deep learning. The first additional variable that Adam keeps track of is the exponential moving average of the first moment (first moment being the gradient):

<figure>
  <img src="blogs/basic-nn-figures/m_t.png" alt="m_t" />
  <figcaption>equation for the first moment</figcaption>
</figure>

The idea is that gradients are noisy, especially with mini-batches rather than the full datase, so instead of descending by the raw gradient, we track a smoothed average of past gradients and scale each parameter's step size by how large its gradients have historically been. The second variable to keep track of is the exponential moving average of the second moment (second moment being the gradient squared):

<figure>
  <img src="blogs/basic-nn-figures/v_t.png" alt="v_t" />
  <figcaption>equation for the second moment</figcaption>
</figure>

Due to the nature of the recursive form of these equations and $m_0$ and $v_0$ being set to 0, the equations do not capture the expected value of $m$ nor $v$. As seen in the derivation below, the expected value of $m_t$, $\mathbb{E}[m_t]$, is equal to the $(1 - \beta_1^{t}) \mathbb{E}[g]$, so before the equations are added to the Adam optimizer they must be divided by $(1 - \beta_1^{t}) \mathbb{E}[g]$. 

<figure>
  <img src="blogs/basic-nn-figures/m-unroll.png" alt="unroll m" />
  <figcaption>adjustment to first moment</figcaption>
</figure>

and similarly we adjust $v_t$:

<figure>
  <img src="blogs/basic-nn-figures/v-unroll.png" alt="unroll v" />
  <figcaption>adjustment to second moment</figcaption>
</figure>

With both moments in hand, we update parameters with adam via:

<figure>
  <img src="blogs/basic-nn-figures/adam.png" alt="adam update" />
  <figcaption>Adam update</figcaption>
</figure>


We divide by the square root of the second moment since a large second moment indicates that gradients in this direction have been high magnitude or high variance, so we scale the step down accordingly - either we might overshoot, or the signal is too noisy to trust. A small second moment means small, stable gradients, so we scale the step up. The $\epsilon$ term is set to a very small non-zero number and is there to prevent division by zero when $\hat{v}$ is 0.

The Adam update for each parameter is performed with the following generalized function, which applies the computed gradients and moment estimates to adjust the parameters according to the afforementioned equations. 

```cpp
void nn::adam(Eigen::MatrixXf &theta, Eigen::MatrixXf &m, Eigen::MatrixXf &v, const Eigen::MatrixXf &g) {
  //first moment
  m = beta1 * m + (1.f - beta1) * g;
  Eigen::MatrixXf mAdj = m / (1.f - std::pow(beta1, iter)); //adjust
  
  //second moment
  v = beta2 * v + (1.f - beta2) * g.cwiseProduct(g);
  Eigen::MatrixXf vAdj = v / (1.f - std::pow(beta2, iter)); //adjust
  
  //adam update
  theta.array() -= lr * mAdj.array() / (vAdj.array().sqrt() + eps);
}
```

and we apply this adam update to all the parameters of interest: $w^[1]$, $w^[2]$, $b^[1]$, $b^[2]$
```cpp
void nn::step() {
  iter += 1.0f;
  adam(w1, m_dJdw1, v_dJdw1, dJdw1);
  adam(w2, m_dJdw2, v_dJdw2, dJdw2);
  adam(b1, m_dJdb1, v_dJdb1, dJdb1);
  adam(b2, m_dJdb2, v_dJdb2, dJdb2);
}
```

pyTorch's `step` method is applying one step of the optimizer's update (which you can set to be the Adam optimizer) similarly under the hood:

```python
optimizer.step()
```

### Mini-batch

Up until now, I wasn't explicit about what the m "examples" consist of. The entire dataset? The entire training set? Neither. We choose to set m to a random mini-batch chosen to be 256. Computing a forward pass and backpropagation on a mini-batch is much faster than on the entire dataset, so we get far more gradient updates per unit time. The gradient estimate is noisier, but the volume of updates more than compensates.

In the code to implement stochastic (random) mini-batch for each iteration, each epoch begins by listing every training index `0, 1 … N-1`, then shuffling that list once. Training examples are stored as columns of a matrix `X` (and matching entries in a label vector `y`), so a “mini-batch” really means “take `batchSize` columns at a time.” Shuffling the indices makes each batch a random subset of the training set. The loop over `step` runs once per batch in an epoch: `stepsPerEpoch` is chosen so that `batchSize * stepsPerEpoch` covers the training set for that shuffle.

```cpp
// shuffle indices for mini batches for sgd
std::vector<Eigen::Index> idx(trainN);
std::iota(idx.begin(), idx.end(), 0);
std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});

for (int step = 0; step < stepsPerEpoch; step++) {
    const auto [xBatch, yBatch] = dataLoader(split.xTrain, split.yTrain, 
        batchSize, step, idx); //gets mini batch
    ...
}
```

The `dataLoader` function slices out the correct mini-batch from the full dataset: given the step, shuffled indices (`idx`), and batch size, it picks out which columns of the input matrix `x` and which entries of the label vector `y` should make up the current batch. For each sample in the batch, it copies over the relevant column from `x` and the corresponding label from `y`, using `idx[step * batchSize + i]` to look up the randomized position, returning the smaller `xBatch` matrix and `yBatch` vector that make up this batch.

```cpp
std::pair<Matrix, Eigen::VectorXf> dataLoader(const Matrix &x, 
const Eigen::VectorXf &y, int batchSize, 
int step, const std::vector<Eigen::Index> &idx) {

  Matrix xBatch(x.rows(), batchSize);
  Eigen::VectorXf yBatch(batchSize);
  for (int i = 0; i < batchSize; i++) {
    Eigen::Index col = idx[step * batchSize + i];
    xBatch.col(i) = x.col(col);
    yBatch(i)     = y(col);
  }
  return {xBatch, yBatch};
}
```

## Putting Everything Together

The code is implemented in a class `nn`. This is done to allow for easy access of member variables such as the weights and biases which get constantly updated in forward passes, and accessed in backpropogation. Here's a snippet of the skeleton of the class from the [entire code](https://github.com/mpham8/nn-fundamental/blob/main/nn-torchless/src-cpp/main.cpp).


```cpp
class nn {
 private:
  //model weights
  Eigen::MatrixXf w1;
  Eigen::MatrixXf b1;
  ...

  static Matrix Linear(const Matrix &x, const Matrix &w, const Matrix &b);
  static Matrix ReLu(const Matrix &x);
  static Matrix softmax(const Matrix &x);
  void adam(Eigen::MatrixXf &theta, Eigen::MatrixXf &m, Eigen::MatrixXf &v, const Eigen::MatrixXf &g);


 public:
  //adam optimizer hyperparameters
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  
  
  nn(int inputDim, int hiddenDim, int outputDim) 
  ...

  Matrix forward(const Matrix &x);
  static Matrix oneHotEncoding(const Eigen::VectorXf &y, int numClasses);
  void backward(const Matrix &x, const Eigen::VectorXf &y);
  void step();
  void adamSettings(const float newLR, const std::vector<float> &newBetas, const float newEps);

};
```

The training loop ties everything together. Each epoch (one complete pass through the entire training dataset), we shuffle the dataset indices and iterate over mini-batches. For each mini-batch, `dataLoader` fetches the samples, `model.forward()` computes the intermediate activations and the output, `model.backward()` computes the gradients, and `model.step()` updates the parameters. Accuracy and timing are logged at the end of each epoch.

```cpp
//training loop
for (int epoch = 1; epoch < epochs; epoch++) {

  // shuffle indices for mini batches for sgd
  std::vector<Eigen::Index> idx(trainN);
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});

  for (int step = 0; step < stepsPerEpoch; step++) {
    const auto [xBatch, yBatch] = dataLoader(split.xTrain, split.yTrain, batchSize, step, idx); //gets mini batch

    (void)model.forward(xBatch); //forward propogate
    model.backward(xBatch, yBatch); //backward propogate to get gradients
    model.step(); //parameter update step
  }

  //get accuracy metrics and print
  const int trainAcc = accuracy(model.forward(split.xTrain), split.yTrain);
  const int testAcc = accuracy(model.forward(split.xTest), split.yTest);
  std::cout << "epoch: " << epoch << "/" << epochs 
  << "  train acc: " << trainAcc << "%  test acc: " << testAcc << "%"
  << "  time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - trainStart).count() << "secs\n";
}
  ```

In 20 epochs in 2 seconds on my laptop, we're able to get 96% accuracy on the test set! Great success!

<figure>
  <img src="blogs/basic-nn-figures/training.png" alt="training output" />
  <figcaption>training output</figcaption>
</figure>

