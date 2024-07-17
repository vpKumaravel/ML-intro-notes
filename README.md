# ML-intro-notes

## Chapter 1
### Use of Categorical Features
Make the features categorical rather than distinct values such that search space is optimised. But, of course, the classifier might not reach 100% accuracy.
This can be done by histogram binning, clustering or domain-specific grouping. An example showing how to label age groups, rather than training for individual age values:


```python
import pandas as pd

data = {'Age': [22, 25, 47, 52, 46, 56, 60, 78, 90]}
df = pd.DataFrame(data)

bins = [0, 18, 35, 50, 65, 100]
labels = ['0-18', '19-35', '36-50', '51-65', '66-100']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

### Need for additional or interpretable information
Just the features and class labels are not enough, additional information (or attributes) supporting the class labels would be beneficial. However, there can be different issues related to the attributes. We might provide irrelevant attributes for the classification problem, or provide a redundant attribute (less dangerous), or miss out an important attribute, or miss a value for the attribute or even a wrong value (attribute noise - stochastic/random or systematic/repetitive). Then, we have class label noises - because certain samples can be difficult to label as strictly positive or negative class (ambiguous samples). Also, an expert might label a sample incorrectly. 

It is therefore, a good idea to start doing feature analysis - an exporatory data analysis (EDA) to look into missing values, and identify redundancies, label noises, so on so forth.

## Chapter 2 

### Definitions
Chapter 2 starts with basic definitions for prior probability, conditional probability and joint probability.

- **Prior probability** \( P(A) \) refers to the initial assessment of the likelihood of an event occurring before any additional information is considered. It is the probability that is assigned to an event based on previous knowledge or experience.

- **Conditional probability** is the probability of an event occurring given that another event has already occurred. It provides a way to update the probability of an event based on new information.

The conditional probability of event \( A \) given that event \( B \) has occurred is denoted as \( P(A|B) \). It is defined by the formula:

![Conditional Probability](https://latex.codecogs.com/svg.latex?\Large%20P(A|B)%20=%20\frac{P(A%20\cap%20B)}{P(B)})

Where:
- \( P(A|B) \) is the **conditional probability**: the probability of event \( A \) given that event \( B \) has occurred.
- \( P(A \cap B) \) is the **joint probability**: the probability of both events \( A \) and \( B \) occurring.
- \( P(B) \) is the **marginal probability**: the probability of event \( B \) occurring.

- **Joint probability** refers to the probability of two events occurring simultaneously. It is the likelihood of both events \( A \) and \( B \) happening at the same time.

The joint probability of events \( A \) and \( B \) is denoted as \( P(A \cap B) \).

**Summary**

- **Prior Probability \( P(A) \)**: The initial probability of an event based on prior knowledge.
- **Conditional Probability \( P(A|B) \)**: The probability of an event occurring given that another event has occurred.
- **Joint Probability \( P(A \cap B) \)**: The probability of two events occurring simultaneously.

## Bayes' Theorem

Bayes' Theorem is used to update the probability estimate for a hypothesis based on new evidence.

### Formula

![Bayes' Theorem](https://latex.codecogs.com/svg.latex?\Large%20P(H|E)%20=%20\frac{P(E|H)%20\cdot%20P(H)}{P(E)})

Where:
- \( P(H|E) \) is the **posterior probability**: the probability of hypothesis \( H \) given the evidence \( E \).
- \( P(E|H) \) is the **likelihood**: the probability of evidence \( E \) given that hypothesis \( H \) is true.
- \( P(H) \) is the **prior probability**: the initial probability of hypothesis \( H \) before seeing the evidence.
- \( P(E) \) is the **total probability of evidence**: the probability of observing the evidence under all possible hypotheses.

### Interpretation

1. **Start with Prior**: Begin with an initial guess or belief about the hypothesis.
2. **Evaluate Likelihood**: Assess how likely the observed evidence is, assuming the hypothesis is true.
3. **Update to Posterior**: Combine the prior and the likelihood to get an updated belief (posterior probability

Here, the tricky part is computing the likelihood of an evidence E being in the Hypothesis H. (the likelihood of a sample x being in the class C).
To do this, we need to assume the distribution of all samples X for each class C. Naive Bayes model assumes that each attribute (features) are independent, therefore, 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(X|C)&space;=&space;\prod_{i=1}^{n}&space;P(X_i|C)" title="P(X|C) = \prod_{i=1}^{n} P(X_i|C)" />


However, this assumption of attributes being independent is not always true (e.g., when the weight of an object grows, its size grows, too). In such cases, considering Gaussian (bell curve) distribution or Bernoulli distribution for the data might be reasonable.

## Chapter 3 - Similarities: Nearest-Neighbor Classifiers

Rather than real objects such as a cat or a dog, the ML classifier looks for similarities in the attributes to make a prediction. If two instances/samples are close to each other in terms of, say, Euclidean distance, they more likely belong to the same class of objects. Having a single neighbor to decide can be sensitive to noise. To make the classifier more robust, we go for _k_ nearest neighbors. For a binary class problem, _k_ can be an odd number (e.g., 3) to avoid ties.

Objects are similar if the geometric distance between the vectors describing them is small.

### Measuring similarity

An option is to use Euclidean distance. Hamming distance is useful when dealing with Boolean domains. When you have mixed continuous and discrete attributes, care must be given to avoid issues like _scaling_, which will be described in detail using two concepts:

- Irrelevant Attributes: It is crucial to consider how the object is represented in the vector space. Not all attributes are equally important. Some might be _irrelevant_. However, such attributes can influence the geometric distance.
- Scale of Attributes: Large range of values of one of the attributes can become a dominating factor. So, it is essential to normalize the values to be in the range [0,1]

However, a potential downside of normalization is that the values falling between 0 and 1 may not be justified for some attributes. It requires little experimentation along with domain insights.

## Reference

```bibtex
@book{Kubat2015,
  title = {An Introduction to Machine Learning},
  ISBN = {9783319200101},
  url = {http://dx.doi.org/10.1007/978-3-319-20010-1},
  DOI = {10.1007/978-3-319-20010-1},
  publisher = {Springer International Publishing},
  author = {Kubat, Miroslav},
  year = {2015}
}
```
