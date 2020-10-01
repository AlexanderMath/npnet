import numpy as np
import matplotlib.pyplot as plt

def relu(X): 			return X  * (X > 0)

class NumpyNet(): 
	def __init__(self, 	arch		= [28**2, 28**2, 28**2, 28**2], 
						step_size	= 0.0001, 
						momentum	= 0.0,
						batch_size  = 100, 
						plot		= None, 
						epochs		= 10, 
						save_name	= "model.pkl"):

		self.step_size 		= step_size
		self.batch_size 	= batch_size
		self.plot 			= plot
		self.epochs			= epochs
		self.loss_ 			= []
		self.val_loss_ 		= []
		self.save_name		= save_name
		self.momentum		= momentum
		self.arch 			= arch
		self.count 			= 0 

		# Xavier Initialization. 
		v			= np.sqrt(6. / (arch[0] + arch[1]))
		self.W1		= np.random.uniform( -v, v, (arch[0], arch[1]))
		self.b1		= np.random.uniform( -v, v, arch[1])

		v			= np.sqrt(6. / (arch[1] + arch[2]))
		self.W2		= np.random.uniform( -v, v, (arch[1], arch[2]))
		self.b2		= np.random.uniform( -v, v, arch[2])

		v			= np.sqrt(6. / (arch[2] + arch[3]))
		self.W3		= np.random.uniform( -v, v, (arch[2], arch[3]))
		self.b3		= np.random.uniform( -v, v, arch[3])

		if momentum != 0: 
			self.M1 	= np.zeros(self.W1.shape)
			self.M2 	= np.zeros(self.W2.shape)
			self.M3 	= np.zeros(self.W3.shape)
		else: 
			self.M1 = self.M2 = self.M3 = None


	def forward(self, X, mult=None): 
		arch = self.arch
		x0 = X
		x1 = relu(x0 @ self.W1 + self.b1)
		x2 = relu(x1 @ self.W2 + self.b2 )
		pred = x2 @ self.W3 + self.b3
		return pred

	def error(self, X, y): return np.mean((self.forward(X) - y)**2)/2

	def fit(self, X, y, Xval=None, yval=None): 
		n, d = X.shape

		epochs 		= self.epochs
		batch_size 	= self.batch_size
		arch		= self.arch

		for epoch in range(self.epochs): 
			# Shuffle data. 
			p 			= np.random.permutation(n)
			X 			= X[p]
			y 			= y[p]
			acum_loos 	= 0.0

			# Loop for mini-batches potentially disregarding last batch. 
			for batch_num in range(n // self.batch_size): 
				batch 		= slice(batch_num*batch_size, (batch_num+1)*batch_size)
				current_X	= X[batch] 
				current_y	= y[batch] 

				batch_loss = self.step(current_X, current_y)

				acum_loos += batch_loss

				print("\r%i / %i\tLoss: %.4f"%((batch_num+1)*batch_size, n, batch_loss), end="")
				if not self.plot is None: self.plot(self, mini_batch=True)

			loss = acum_loos / (X.shape[0]/batch_size)
			self.loss_.append(loss)

			# Compute validation error and call plot function if enabled. 
			if not Xval is None: 		self.val_loss_.append(self.error(Xval, yval))
			if not self.plot is None: 	self.plot(self, mini_batch=False)

			# Print progress. 
			if not Xval is None: 	print("Iteration %i: \t%.4f : %.4f"%(epoch+1, self.val_loss_[-1], loss))
			else: 					print("Iteration %i: \t%.4f"%(epoch+1, loss))

			# Save model. 
			self.save(self.save_name + "_epoch%i"%epoch)


	def step(self, X, y):
		n, d 		= X.shape
		step_size 	= self.step_size
		batch_size 	= self.batch_size
		momentum	= self.momentum

		# Forward Pass: 
		a0 = X
		a1 = relu(a0 @ self.W1 + self.b1)
		a2 = relu(a1 @ self.W2 + self.b2)
		a3 = 	  a2 @ self.W3 + self.b3 # no output activation function. 
		
		error = np.mean((a3-y)**2) / 2

		# Backwards Pass: 
		# True for identity + squared error, sigmoid+BCE, softmax+categorical cross entropy.
		d3 = (a3-y) 
		d2 = (d3 @ self.W3.T) 
		d1 = (d2 @ self.W2.T)

		d2[a2 == 0] = 0 # compute derivative of relu activations inplace inspired by sklearn.
		d1[a1 == 0] = 0

		# last part is (100 x 3) and (100 x 4)
		dw3 = a2.T @ d3 
		dw2 = a1.T @ d2 
		dw1 = a0.T @ d1 

		# Update weights and biases

		# Momentum part, for simplicity disregards momentum on bias. 
		if momentum == 0: 
			self.W3 -= step_size * dw3 / batch_size
			self.W2 -= step_size * dw2 / batch_size
			self.W1 -= step_size * dw1 / batch_size
		else: 
			self.M3 = momentum * self.M3 + dw3
			self.M2 = momentum * self.M2 + dw2
			self.M1 = momentum * self.M1 + dw1

			self.W3 -= step_size * self.M3 / batch_size
			self.W2 -= step_size * self.M2 / batch_size
			self.W1 -= step_size * self.M1 / batch_size

		self.b3 -= step_size * np.mean(d3, axis=0)
		self.b2 -= step_size * np.mean(d2, axis=0)
		self.b1 -= step_size * np.mean(d1, axis=0)

		return error


	def save(self, name): 	
		import pickle
		temp = self.plot 
		self.plot = None
		with open("models/" + name, 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
		self.plot = temp

	def load(name): 		
		import pickle
		with open("models/" + name, 'rb') as handle: nn = pickle.load(handle)
		return nn


# Plot code.
plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(8, 3))

def plot(nn, mini_batch):
	ax[0].cla()
	ax[1].cla()
	ax[1].axis('off')
	ax[2].axis('off')

	# plot cost
	xs = range(1, len(nn.loss_)+1)
	ax[0].plot(xs, nn.loss_, 		label="Training error")
	ax[0].plot(xs, nn.val_loss_, 	label="Validation error")
	ax[0].set_title("Loss")
	ax[0].set_xlim([0, nn.epochs])
	ax[0].set_ylim([0, 0.05])
	ax[0].set_ylabel("Error")
	ax[0].set_xlabel("Epochs")
	ax[0].legend()

	# reconstruction
	ax[1].imshow(X[0].reshape(28, 28), vmin=0, vmax=1, cmap="gray") # real image. 
	ax[1].set_title("Real Digit")
	rec = nn.forward(X[0].reshape(1, 28**2))
	ax[2].set_title("Reconstruction")
	ax[2].imshow(rec.reshape(28, 28), vmin=0, vmax=1, cmap="gray")

	# make plot. 
	plt.tight_layout()
	plt.pause(0.1)

	# save for gif. 
	plt.savefig("imgs/%i.jpg"%nn.count)
	nn.count += 1 


if __name__ == "__main__":

	# Load and normalize MNIST.
	from keras.datasets import mnist
	(X, y), (X_test, y_test) = mnist.load_data()
	X 		= X.reshape(X.shape[0], np.prod(X.shape[1:]))
	X_test  = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
	X 		= X/2**8
	X_test  = X_test/2**8


	# Hyper parameters. 
	step_size 	= 0.01
	d 			= 28**2
	hu 			= d//2
	batch_size 	= 100 
	epochs 		= 10
	arch		= [d, hu, hu, d]

	# Initialize and train neural network. 
	nn 	= NumpyNet(step_size=step_size, batch_size=batch_size, arch=arch, plot=plot, epochs=epochs)
	nn.fit(X[:1000], X[:1000], Xval=X[1000:1100], yval=X[1000:1100]) 
	plt.show() 

