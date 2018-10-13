function [G] = grbf_fast(Data,center,Diag_S);
% GRBF_FAST calculates the design matrix G(# of data, # of centers). Auckland 2002.
% GRBF_FAST is THE FASTET version FOR GRBF in MATLAB for a diagonal covariance matrix. 
% GAUSSIANS MAY HAVE SAME or DIFFERENT VARIANCES !!!
% Written by Vojislav Kecman - at The University of Auckland, New Zealand & VCU, Richmond, VA, 
% First versions written in Yugoslavia and Germany in 1990-ties
% 	Program description:
% 	Inputs:		Data, centers and s must be of same dimensionality
% 		Data		  Input pattern for training 		
%			center		Matrix of the Gaussian centers
%   		Diag_S	= 1./vector of sigmas.  NOTE!!! It is not the inverse of covariance matrix. 
%	Output:  matrix G, 				Copyright (c) 2002-2018 by Vojislav KECMAN

x = Diag_S*Data';	c = Diag_S*center';

xx = sum(x.*x,1); 		cc = sum(c.*c,1); 		xc = x'*c; 
d = (repmat(xx',[1 size(cc,2)]) + repmat(cc,[size(xx,2) 1]) - 2*xc);		

G = exp(-0.5*d);



