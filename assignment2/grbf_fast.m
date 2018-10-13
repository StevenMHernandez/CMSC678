function [G] = grbf_fast(Data,center,s);
% GRBF_FAST calculates the design matrix G(# of data, # of centers). Auckland 2002.
% GRBF_FAST is one of THE FASTET version FOR GRBF in MATLAB for calculation of Gaussian kernel matrix G when all Gaussians have same s 
% Written by Vojislav Kecman - at The University of Auckland, New Zealand & VCU, Richmond, VA, 
% First versions written in Yugoslavia and Germany in 1990-ties
% 	Program description:
% 	Inputs:		Data and centers must be of same dimensionality
% 		Data		  Input pattern for training or testing		
%			center		Matrix of the Gaussian centers
%   		s = sigma i.e., s = std of Gaussian kernel . s is A SCALAR
%	Output:  matrix G, 				Copyright (c) 2002-2018 by Vojislav KECMAN

x = Data'/s;	c = center'/s;

xx = sum(x.*x,1); 		cc = sum(c.*c,1); 		xc = x'*c; 
d = (repmat(xx',[1 size(cc,2)]) + repmat(cc,[size(xx,2) 1]) - 2*xc);		

G = exp(-0.5*d);



