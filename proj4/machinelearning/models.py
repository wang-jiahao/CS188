import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 2 * (nn.as_scalar(self.run(x)) >= 0) - 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            errors = 0
            for x, y in dataset.iterate_once(1):
                p = self.get_prediction(x)
                s = nn.as_scalar(y)
                if p != s:
                    errors += 1
                    (self.w).update(x, s)
            if errors == 0:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.lr = 0.05
        self.bs = 200
        self.m1 = nn.Parameter(1, 512)  # shape x by 512
        self.b1 = nn.Parameter(1, 512)
        self.b2 = nn.Parameter(1, 1)
        self.m3 = nn.Parameter(512, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # x--->m1 --->ReLU--->m2--->pred
        a = nn.Linear(x, self.m1)
        b = nn.AddBias(a, self.b1)
        c = nn.ReLU(b)
        d = nn.Linear(c, self.m3)
        pred = nn.AddBias(d, self.b2)
        return pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        i = 0
        for x, y in dataset.iterate_forever(self.bs):
            i += 1
            gradm1, gradm3, gradb1, gradb2 = nn.gradients(self.get_loss(x, y), [self.m1, self.m3, self.b1, self.b2])
            multiplier = self.lr
            self.m1.update(gradm1, -multiplier)
            self.m3.update(gradm3, -multiplier)
            self.b1.update(gradb1, -multiplier)
            self.b2.update(gradb2, -multiplier)
            acc = nn.as_scalar(self.get_loss(x, y))
            if i % 100 == 0:
                print(acc)
            if acc < 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        self.lr = 0.5
        self.bs = 100
        self.m1 = nn.Parameter(784, 200)  # shape x by 512
        self.b1 = nn.Parameter(1, 200)
        self.m3 = nn.Parameter(200, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        a = nn.Linear(x, self.m1)
        b = nn.AddBias(a, self.b1)
        c = nn.ReLU(b)
        d = nn.Linear(c, self.m3)
        pred = nn.AddBias(d, self.b2)
        return pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        i = 0
        for x, y in dataset.iterate_forever(self.bs):
            i += 1
            gradm1, gradm3, gradb1, gradb2 = nn.gradients(self.get_loss(x, y), [self.m1, self.m3, self.b1, self.b2])
            multiplier = self.lr
            self.m1.update(gradm1, -multiplier)
            self.m3.update(gradm3, -multiplier)
            self.b1.update(gradb1, -multiplier)
            self.b2.update(gradb2, -multiplier)
            acc = dataset.get_validation_accuracy()
            if i % 100 == 0:
                print(acc)
            if acc > 0.98:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.lr = 0.3
        self.bs = 200
        self.m1 = nn.Parameter(self.num_chars, 200)
        self.m2 = nn.Parameter(200, len(self.languages))
        self.m3 = nn.Parameter(200, 200)
        self.m4 = nn.Parameter(len(self.languages), 200)
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, len(self.languages))
        self.b3 = nn.Parameter(1, 200)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # [[[c],[b],[h]][[a],[a],[a]][[t],[t],[t]]]
        x = xs[0]  # first characters
        x = nn.Linear(x, self.m1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.m3)
        x = nn.AddBias(x, self.b3)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.m2)
        h = nn.AddBias(x, self.b2)

        for i in range(1, len(xs)):
            x = xs[i]
            x = nn.Linear(x, self.m1)
            h = nn.Linear(h, self.m4)
            x = nn.Add(h, x)
            x = nn.AddBias(x, self.b1)
            x = nn.ReLU(x)
            x = nn.Linear(x, self.m2)
            h = nn.AddBias(x, self.b2)

        return h

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        i = 0
        for xs, y in dataset.iterate_forever(self.bs):
            i += 1
            gradm1, gradm2, gradm3, gradm4, gradb1, gradb2, gradb3 = nn.gradients( \
                self.get_loss(xs, y), [self.m1, self.m2, self.m3, self.m4, self.b1, self.b2, self.b3])
            multiplier = self.lr / (i ** 0.3)
            self.m1.update(gradm1, -multiplier)
            self.m2.update(gradm2, -multiplier)
            self.m3.update(gradm3, -multiplier)
            self.m4.update(gradm4, -multiplier)
            self.b1.update(gradb1, -multiplier)
            self.b2.update(gradb2, -multiplier)
            self.b3.update(gradb3, -multiplier)
            acc = dataset.get_validation_accuracy()
            if (i % 100 == 0):
                print("\n" + str(i) + ": " + str(multiplier) + "\n")
            if acc > 0.84:
                break