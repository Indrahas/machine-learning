function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

a=X*theta;
b=sigmoid(a);
c=log(b);
d=log(1-b);
e=y.*c;
f=1-y;
g=f.*d;
h=e+g;
J=(sum(h)*-1)/m;
[A,B]=size(X);
l=2:B;
for n=l,
o=theta(n)*theta(n);
p=(lambda*o)/(2*m);
J=J+p;
end


i=b-y;
j=X.*i;
k=sum(j);
grad=k'./m;
for n=l,
grad(n)=grad(n)+((lambda*theta(n))/m);
end




% =============================================================

end
