# ML-intro-notes

## Use of Categorical Features
Make the features categorical rather than distinct values such that search space is optimised. But, of course, the classifier might not reach 100% accuracy.
This can be done by histogram binning, clustering or domain-specific grouping. An example 


```python
import pandas as pd

data = {'Age': [22, 25, 47, 52, 46, 56, 60, 78, 90]}
df = pd.DataFrame(data)

bins = [0, 18, 35, 50, 65, 100]
labels = ['0-18', '19-35', '36-50', '51-65', '66-100']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

'''

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
