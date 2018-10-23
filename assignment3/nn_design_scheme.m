function [] = your_name_Project3
close all, clear all, format compact
seed0=1;	randn('seed',seed0), rand('seed',seed0)

load your data

plot data inputs
plot outputs as circle to 
scale the input if you think it is needed
create train and test set
find numb of train data and call in ntrain, same for ntest

N0 = [5 10 15 25 50 75 100] or, whatever you prefer
I0 = [100 250 500 1000] or if 1000 is not enough go for more

for n = 1:length(N0)
		num_n = N0(n)	
		define random initial HL weighs V, and random OL weights W
    for i = 1:length(I0)  i is an index of an epoch or a sweep through all data
			for j = 1:ntrain
			input is X(j,,:)
				here comes your learning code which basically implements the algorithm as given in the table and example

				you take your first data point and    
	    	here you calculate inputs to HL neurons, their outputs and derivatives of AF at each neuron
	    	input(s) to OL neuron and its output
	    	error_at OL neuron for a given input data
	    	
	    	EBP part comes below now
	    	
	    	delta signal for OL neuron
	    	delta signals for HL neurons
	    	
	    	update OL and HL weights
	    end
    end
    Training is over
    here comes calculation of the error on the test data
    give them, find outputs see errors and save in error matrix E
    E(n,i) = 
end

find best numb of neurons and best numb of iterations.
plotting etc


% in real life you would now go an and design (i.e., retrain) your multilayer perceptron on all the data by using the best numbers
% and your NN is designed



