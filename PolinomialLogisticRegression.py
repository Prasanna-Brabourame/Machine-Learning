"""
This code shows an implementation of polinomial logistic regression, normalization,
automatic alpha tuning, convergence of training algorithm, regularization to avoid
overfitting.

We changed from np.matrix to np.array to be faster and more flexible, so
instead of using * as matrix dot product now we need to use @
"""
polyorder=6 # here you decide what polinomial order your system will have
lambd = 0.001 # that's the regularization term, higher values make the boundary shape "rounder" (bias), lower values make it "crispier" (overfitting). Reasonable values to try here would be [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, ..., 0]
example=2 # here you choose the example to be used

# suppress the warnings when logarithm calculates infinity
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
print('Welcome to Machine Learning with Python!')
print('Lesson 3: Polinomial Logistic regression')
print('\n'+40*'=')

print('The task now is to create a model which would separate 2 classes of objects based on 2 inputs.\n')
print('But a simple line is not able to perform a good separation because the ', end='')
print('classes are mixed into each other')
# data representing the measurements and category 1 and 2 (or 'o' and 'x')
data1 =\
[(10, 12, 1), (15, 14, 1), (6, 18, 1), (12, 10, 1),\
(17, 13, 1), (18, 18, 1), (11, 15, 1), (3, 17, 1),\
(5, 15, 1), (16, 9, 2), (16, 16, 1), (14, 7, 2),\
(22, 18, 1), (25, 19, 1), (27, 17, 1), (26, 10, 1),\
(18, 12, 1), (24, 7, 1), (23, 12, 1), (28, 9, 1),\
(7, 2, 2), (4, 12, 2), (2, 3, 2), (14, 1, 2),\
(7, 5, 2), (20, 1, 1), (2, 9, 2), (10, 4, 2),\
(2, 7, 2), (4, 1, 2), (13, 2, 2), (10, 1, 2),\
(4, 5, 2), (17, 3, 2), (3, 13, 2), (25, 3, 1),\
(5, 3, 2), (14, 4, 2), (23, 2, 1), (17, 1, 2)]

# second data set so we can choose which example to use
data2 =\
[(15, 10, 1), (7, 9, 1), (10, 12, 2), (18, 8, 1),\
(19, 11, 1), (20, 13, 1), (15, 7, 1), (13, 17, 1), (18, 18, 1),\
(3, 9, 2), (28, 6, 2), (27, 12, 2), (5, 5, 2),\
(10, 4, 2), (26, 4, 2), (8, 16, 2), (22, 16, 2)]

if example==1: data = data1
else: data = data2

dataX1 = np.array(data, dtype=np.float64)[:,0] #float64 is necessary to avoid "wrap around" phenomena in higher orders (>7)
dataX2 = np.array(data, dtype=np.float64)[:,1]
y = 2 - np.array(data, dtype=np.float64)[:,2][np.newaxis].T #is float64 so everything is in the same basis and calculations are faster
	
'''
Extending the Logistic regression model, we want to build a high order
polinomial model able to separate 2 classes of objects based on two inputs

X matrix will hold the inputs - upscaled to a polinomial degree
X has each data point as a row, and has m rows.
The columns are [ constant, X2, X1, X1*X2, X2^2, X1^2] for order 2
[cte, X2, X1, X1*X2, X1*X2^2, X2^2, X1^2, X1^2*X2, X2^3, X1^3 ] for order 3
and so on.

The higher the order the more complex can the classification boundary be, but
computational effort also sky rocks.
'''
def polyX(dataX1, dataX2, pord):
	pord+=1
	X = np.ones((dataX1.shape[0], round(pord/2*(pord+1))), dtype=np.float64)
	k=1
	for i in range(1,pord):
		X[:,k]=dataX2**i
		k+=1
		for j in range(pord-i):
			X[:,k]=(dataX1**i*dataX2**j)
			k+=1
	return X

def normalize(X):
	''' the normalizer puts all X inputs in about the same scale, so
			gradient descent can work better. Only the first collumn
			(the constant) remains at 1'''
	reg=np.ones((2, X.shape[1]))
	reg[0,:]=X.mean(axis=0)
	reg[1,:]=X.std(axis=0)
	reg[0,0]=0
	reg[1,0]=1
	return (X-reg[0])/reg[1], reg # return the adjustment also, so we can readjust further inputs when predicting a value

X = polyX(dataX1, dataX2, polyorder)

print('So we need to choose our degree of polinomial extension, in this case it is {}'.format(polyorder))
print('this means our new X input that had a size of 2 (or 3 with the constant) now has {} elements for each example\n'.format(X.shape[1]))
print('X[0] (for the first point) = '+np.array_str(X[0], precision=0, suppress_small=True))
print('as inputs numbers get orders of magnitute to each other (3 compared to 6546778) a normalization of the input data is necessary. This can be done using average and standard deviation, so every new X has an average around zero and a span of about +/- 3')
[X, Xnor] = normalize(X)
print('X[0] after normalization   = '+np.array_str(X[0], precision=3, suppress_small=True))

'''
Below is the show() function, which will use the ASCII to graphically "plot" the charts
'''

def show(arr, p):
	for j in range(np.size(arr,1)-1, -1, -1):
		print('{:>2}'.format(j if j%5 == 0 else ' ')+' '+chr(179+(j%5==0)), end='')
		for i in range(np.size(arr,0)):
			print(p[arr[i][j]], end='')
		print()
	print('   '+chr(192)+(chr(194)+chr(196)*4)*(np.size(arr,0)//5))
	print('	', end='')
	print(*list(('{:<4}'.format(i*5) for i in range(np.size(arr,0)//5))))
	print()

# t and p are the plot variables
t = np.zeros((30,20), np.int8)
p = [' ', '#', 'o', 'x']

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def J(X, y, theta, lambd):
	# X is m samples x j features, y is m samples x k outputs, theta is j features x k outputs
	# all three arguments must be given as np.array already in the right shape
	m = y.shape[0] # m is the number of datapoints
	predictions = sigmoid(X@theta) # stores the sigmoid predictions, translated into "binary" classes
	l0 = np.log(predictions)
	l1 = np.log(1-predictions)
	l0[l0==-np.inf] = -100
	l0[l0==np.inf] = -100
	l1[l1==-np.inf] = -100
	l1[l1==np.inf] = -100
	# here we included lambda*sum(theta) as the regularization term, 
		# that penalizes the error with increasing values of theta.
	return 1/m * (((-y).T@l0) - ((1-y).T@l1) + (lambd/2*np.sum(theta[1:,:])))[0,0], 1/m * (X.T @ (predictions-y)) # the second parameter is the gradient


'''
The optimal theta vector will be needed for our model to predict with the smallest possible error.
Assuming that J is a cost function, this is again an optimization problem - we need to find the minimum of J.
The gradient descent function looks somewhat as the logistic regression.
This time, however, we have automatic alpha, regularization (to keep theta values
low and avoid overfitting) and some calculation enhancements so the code runs faster.
'''

# gradient descent function will iteratively update theta by a small fraction alpha (also called the learning rate) for a number of iterations
def gradient(X, y, theta, iters, lambd):
	iters+=1
	m = len(y) # m is the number of datapoints
	alpha = 0.01 # initial learning rate, will be adjusted further in the loop
	J_history = np.zeros(iters, dtype=np.float64) # will store historical values of J for each iteration
	reg = np.ones((len(theta),1), dtype=np.float64)
	[J_history[0], delta] = J(X, y, theta, lambd) # get the first data set
	for i in range(1,iters):
		oldtheta = theta # remember last values
		olddelta = delta
		reg.fill(1-alpha*lambd/m) # this is the regularization term
		reg[0] = 1  # no regularization for the constant
		theta = (theta * reg) - (alpha * delta) # update theta by learning rate times gradient
		[J_history[i], delta] = J(X, y, theta, lambd) # save the cost J of a particular iteration and get the slopes for adjusting in the next loop
		if J_history[i] <= J_history[i-1]: #  It should drop in the next iteration, if not alpha is too big and must be lowered
			alpha *= 1.05 # increase alpha slowly to get a faster optimum
		else:
			alpha *= 0.3 # decrease alpha rapidly, we tried to be too fast and J increased
			theta = oldtheta # restore the old values for gradient descend track from there
			delta = olddelta
			J_history[i]=J_history[i-1]
			if (alpha*np.max(np.abs(delta/theta))<5e-4): # local minimum has been reached because no significant change will happend in the next iteration - increase the number if "Time limit exceeded"
				print('converged after {} loops.'.format(i), end='')
				break
	return J_history, theta # return the history of cost J (for debugging) plus the optimal theta

print('\n'+40*'=')
print('Using a higher order polynomial input can lead to overfitting, that might reflect negatively in the model prediction ability with new unknowen data. To try to avoid this we introduce a regularization term Lambda that tries to keep theta values low. This has an "rounding" effect on the decision boundary')
theta = np.zeros((X.shape[1],1), dtype=np.float64)
iters = 50000 # number of maximum iterations - reduce if "Time limit exceeded"
print('\n== Model summary ==\nMax. iterations: {}\nLambda: {}\nInitial theta:\n{}\nInitial J: {:.12f}\n'.format(iters, lambd, theta.T, J(X,y,theta,lambd)[0].item()))

print('Training the model... ', end='')
J_history, theta_min = gradient(X, y, theta, iters, lambd)
print('Done.')

print('\nFinal theta:\n'+np.array_str(theta_min.T, precision=3, suppress_small=True)+'\nFinal J: {:.12f}\n'.format(J(X,y,theta_min,lambd)[0].item()))

'''
Now that we have the model trained, let's check if it works. We will plot the chart with the decision boundary calculated by the model and on top of that, we will plot the real datapoints (our training set). To do that, we will iteratively cast a 'predict' function (defined below) on every possible measurement pair.
'''

def predict(a,b,nor,porder,theta_):
	a=np.array([[a]], dtype=np.float64)
	b=np.array([[b]], dtype=np.float64)
	Xpred=(polyX(a,b,porder)-nor[0])/nor[1]#here we are normalizing the input using the same values as in the beginning
	return np.round(sigmoid(Xpred@theta_))
# round does the differenciation that if h0>0,5 then prediction is 1, otherwise it is 0

print('\n'+40*'=')

# db will store the chart for displaying the decision boundary
db = np.zeros((30,20), dtype=np.int8)
for j in range(np.size(db, 1)):
	for i in range(np.size(db, 0)):
		db[i][j] = predict(i,j,Xnor,polyorder,theta_min)[0,0]

# Populating the plot array from the measurement data
for i in data:
	db[i[0]][i[1]] = i[2]+1

# let's show it now
print('\nDoing predictions...\nEverything in the dotted area will be predicted by the model as a "o" element, while datapoints belonging to the blank area will be predicted as "x" ones.')
show(db, [' ', '.', 'o', 'x'])

def score(X, y, theta):
	return np.mean(np.round(sigmoid((X)@theta))==y), np.sum(np.round(sigmoid((X)@theta))==y)

print('Model accuracy {:.1%} ({}/{} points)'.format(score(X, y, theta_min)[0], score(X, y, theta_min)[1], len(y)))

print('Feel free to modify the datapoints, lambda or the polinomial depth and see what happens! :)')
