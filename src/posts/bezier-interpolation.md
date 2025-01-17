---
title: Bezier Curve for Imputation
desc: My little, for fun library project to impute missing value using Bezier curve
date: '2025-01-14'
tags:
    - Numeric
    - Machine Learning
published: true
---

## Introduction to Bezier Curve
Bezier curve is a parametric curve that usually used in computer graphics. It is defined by $n+1$ control points $\mathbf{P} = [P_0,P_1,\dots,P_n]$, where $n$ is the order of Bezier Curve, which dictate the shape of the curve. Mathematically, Bezier curve can be expressed explicitly as follows:
$$
B(t;\mathbf{P}) = \sum_{i=0}^{n}b_{i,n}(t)P_i,\text{ }0 \leq t \leq 1
$$
where
$$
b_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i},\text{ }i = 0,1,\dots,n
$$
$$
\binom{n}{i} = \frac{n!}{i!(n-i)!}
$$

## The Idea
It got me thinking, is it possible to use Bezier curve formula to impute missing value? Let $\mathbf{y} = [y_1,y_2,\dots,y_m]$ is data with missing values, I start with generating equally spaced $\mathbf{t} = [t_1,t_2,\dots,t_m], 0 \leq t_j \leq 1$ with $m$ is number of data samples

```python
import numpy as np

t = np.linspace(0,1,len(data))
```

then approximate the data $\mathbf{y}$ with Bezier curve formula

$$
\mathbf{\hat{y}} = [B(t_1;\mathbf{P}),B(t_2;\mathbf{P}),\dots,B(t_m;\mathbf{P})].
$$

But then, how to find control points $\mathbf{P}$ so that the distance between approximation $\mathbf{\hat{y}}$ and real data $\mathbf{y}$ is minimal? For this I'm using Least Square to minimize sum of squared error between approximation and real data, here is a bit of mathematical detail

$$
\nabla_{\mathbf{P}} SSE = 0
$$

$$
\frac{1}{2} \nabla_{\mathbf{P}} (\mathbf{y}-\mathbf{\hat{y}})^T(\mathbf{y}-\mathbf{\hat{y}}) = 0
$$

$$
\frac{1}{2} \nabla_{\mathbf{P}} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{\hat{y}}+ \mathbf{\hat{y}}^T\mathbf{\hat{y}}) = 0
$$

$$
\frac{1}{2} \nabla_{\mathbf{P}} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{A}\mathbf{P}+ (\mathbf{A}\mathbf{P})^T\mathbf{A}\mathbf{P}) = 0
$$

$$
\frac{1}{2} \nabla_{\mathbf{P}} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{A}\mathbf{P}+ \mathbf{P}^T \mathbf{A}^T\mathbf{A}\mathbf{P}) = 0
$$

$$
\frac{1}{2} (-2\mathbf{y}^T\mathbf{A} + 2\mathbf{P}^T \mathbf{A}^T\mathbf{A}) = 0
$$

$$
\mathbf{P}^T \mathbf{A}^T\mathbf{A} = \mathbf{y}^T\mathbf{A}
$$

$$
\mathbf{P}^T = \mathbf{y}^T\mathbf{A} (\mathbf{A}^T\mathbf{A})^{-1}
$$

$$
\mathbf{P} = (\mathbf{A}^T\mathbf{A})^{-1} \mathbf{A}^T\mathbf{y} 
$$

where

$$
\mathbf{A} = 
\begin{bmatrix}
b_{0,n}(t_1) & b_{1,n}(t_1) & \dots & b_{n,n}(t_1)\\
b_{0,n}(t_2) & b_{1,n}(t_2) & \dots & b_{n,n}(t_2)\\
\dots & \dots & \dots & \dots\\
b_{0,n}(t_m) & b_{1,n}(t_m) & \dots & b_{n,n}(t_m)\\
\end{bmatrix},

\mathbf{P} = 
\begin{bmatrix}
P_0\\
P_1\\
\vdots\\
P_n
\end{bmatrix},
$$

$$
b_{i,n}(t_j) = \binom{n}{i}(t_j)^i(1-t_j)^{n-i}.
$$
```python
def bernstein_polynomial(index:int,degree:int,t:float):
    return binomial_coefficient(degree,index) * t**index * (1-t)**(degree-index)

def bezier_gradient(degree:int,t:np.ndarray):
    return np.array(
        [
            [
                bernstein_polynomial(i,degree,t[j])
                for i in range(degree+1)
            ]
            for j in range(len(t))
        ]
    ).T


def least_square_fit(data:np.ndarray,t:np.ndarray,degree:int):
    A = bezier_gradient(degree,t)
    control_points = np.linalg.pinv(A@A.T) @ (A @ data)
    return control_points
```

But there is a problem, data $\mathbf{y}$ has some missing values, we can't directly use Least Square to find control points $\mathbf{P}$. To handle this, I'm just using non-missing values $y_k$ from $\mathbf{y}$ and corresponding $t_k$ from $\mathbf{t}$ to find control points $\mathbf{P}$,

```python
import pandas as pd

nan_mask = data.isna().values
filled = data[~nan_mask].values
control_points = least_square_fit(filled,t[~nan_mask],degree)
```
After finding control points, We impute the missing value using Bezier curve with the estimated control points

```python
import pandas as pd

res = pd.Series(bezier_curve(control_points,degree,t),index=data.index,name=data.name)
data = data.fillna(res)
```

## Result and Possible Future

I tried this on weather data fetched from [Open-Meteo](https://open-meteo.com/), mask 30% of the data randomly and impute it with this method. Mean Absolute Percentage Error (MAPE) between interpolated data and real data before masking is around 1.6%, you can try this method using the [library](https://pypi.org/project/BezierInterpolate/) created by me. 

![Bezier Interpolation](Bezier-Interpolation-Result.PNG)
*Bezier Interpolation Example*

There are some things i want to add in the future, like for example, add [Polars](https://pola.rs/) support to reach more peoples. I also want to add multi-variable support but I still don't know how to efficiently implement it since it needs to estimate control points for all variables or columns. If you want to throw some issue or maybe want to contribute, you can visit [BezierInterpolate](https://github.com/HabilAmardias/BezierInterpolate) repository.

Thank you for reading this article and see you next time :)
