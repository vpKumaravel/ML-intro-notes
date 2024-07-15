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

I think it is therefore, a good idea to start doing feature analysis - an exporatory data analysis (EDA) to look into missing values, and identify redundancies, label noises, so on so forth.

### 

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
