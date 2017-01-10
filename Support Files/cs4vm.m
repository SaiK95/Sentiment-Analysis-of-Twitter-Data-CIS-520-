function [label,prediction,cpu_time] = cs4vm(y,x,x_test,opt)

% cs4vm implements the cs4vm algorithm as shown in [1].
%
%: cs4vm_iter employs the Matlab version of libsvm [2] (available at
%  http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and a slight modified matlab
%  version of libsvm (availaible at 'libsvm-mat-cs4vm' included).
%
%    Syntax
%
%       [label,prediction,cputime] = cs4vm(y,x,x_test,opt)
%
%    Description
%
%       cs4vm takes,
%           y        - A nx1 label vector, where y = 1 means positive, y = -1 means negative, y = 0 means unlabeled 
%           x        - A nxd training data matrix, where d is the dimension of instance 
%           x_test   - A mxd testing data matrix
%           opt      - A structure describes the options of CS4VM
%                      1) opt.c1: regularization term for labeled instances (see Eq.(1) in [1]), default setting opt.c1= 100 
%                      2) opt.c2: regularization term for unlabeled instances(see Eq.(1) in [1]), default setting opt.c2 = 0.1
%                      3) opt.cost: cost for positive class, default setting opt.cost = 1
%                      4) opt.gaussian: kernel type, 1 means gaussian kernel, 0 means linear kernel, default setting opt.gaussian = 0
%                      5) opt.gamma: parameter for gaussian kernel, i.e.,k(x,y) = exp(-gamma*||x-y||^2), default
%                      settting 1/gamma = average distance between patterns
%                      6) opt.maxiter: maximal iteration number, default setting opt.maxiter = 50
%                      7) opt.ep: expected number of positive instances among unlabeled data, default setting opt.ep = prior from labeled data 

%      and returns,
%           label      - A mx1 label vector, the predicted label of the testing data 
%           prediction - A mx1 prediction vector, the prediction of the testing data
%           cputime    - cpu running time
%           
% [1] Y.-F. Li, J. T. Kwok, and Z.-H. Zhou. Cost-Sensitive Semi-supervised Support Vector Machine. In: Proceedings of the 24th AAAI Conference on Artificial Intelligences (AAAI'10), Atlanta, GE, 2010, pp.500-505.
% [2] R.-E. Fan, P.-H. Chen, and C.-J. Lin. Working set selection using second order information for training SVM. Journal of Machine Learning Research 6, 1889-1918, 2005.

tt = cputime;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.1 preprocess the options
n = size(x,1);
m = size(x_test,1);
l_inx = find(y);
u_inx = find(y==0);
ln = length(l_inx);
un = length(u_inx);

if ~isfield(opt,'c1') 
    opt.c1 = 100; 
end
if ~isfield(opt,'c2')
    opt.c2 = 0.1;
end
if ~isfield(opt,'gaussian')
    opt.gaussian = 0;
end
if opt.gaussian == 1 && ~isfield(opt,'gamma')
    opt.gamma = n^2/sum(sum(repmat(sum(x.*x,2)',n,1) + repmat(sum(x.*x,2),1,n) ...
                        - 2*x*x'));
end
if ~isfield(opt,'maxiter')
    opt.maxiter = 50;
end
if ~isfield(opt,'ep')
    opt.ep = ceil(length(find(y == 1))/ln*un);
end
if ~isfield(opt,'cost')
    opt.cost = 1;
end

%1.2 calculate the kernel matrix
if opt.gaussian == 1
    K = exp(- (repmat(sum(x.*x,2)',n,1) + repmat(sum(x.*x,2),1,n)-2*x*x') *opt.gamma);
    K = K + 1e-10*eye(size(K,1));
    K_test = exp(- (repmat(sum(x.*x,2),1,m) + repmat(sum(x_test.*x_test,2)',n,1)- 2*x*x_test') * opt.gamma);
else
    K = x*x';
    K = K + 1e-10*eye(size(K,1));  
    K_test = x*x_test';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. iteratively estimating the label means
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.1 initialize the label means by supervised SVM using label data only

opt_svm = ['-t 4 -c ' num2str(opt.c1) ' w1 ' num2str(opt.cost)];
K_l = K(l_inx,l_inx);
K_l = [(1:ln)',K_l];
addpath('libsvm-mat-2.83-1')
model = svmtrain(y(l_inx),K_l,opt_svm);

K_t = K(l_inx,u_inx);
K_t = [(1:un)',K_t'];
[predict_label, accuracy, dec_values] = svmpredict(ones(un,1),K_t,model);
if model.Label(1) == -1
    dec_values = -dec_values;
end

tmpd = y;
[val,ix] = sort(dec_values,'descend');
tmpd(u_inx(ix(1:opt.ep))) = 1;
tmpd(u_inx(ix(opt.ep+1:un))) = -1;

iter = 1; flag = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.2 fixing label means, solve a SVM problem; fixing SVM, update label means; iteratively until convergence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter < opt.maxiter && flag
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2.2.1 Fixing label means, the SVM problem is a quadratic programming problem whose dual form is as follows,
    % 
    %        max_a  \sum_{i=1}^{ln} a_i - 1/2 (a.*\hat{y})' K^{tmpd} (a.* \hat{y})
    %        s.t.   \sum_{i=1}^{ln} a_i + a_{ln+1} - a_{ln+2} = 0,
    %               a_{ln+1} + a_{ln+2} = c2,
    %               0 <= a_i <= c1, i = 1,..., ln
    %               0 <= a_i <= c2, i = ln+1, ln+2
    % where K^{tmpd} is the kernel matrix and \hat{y} = [y,1,-1];
    % Here, we solve this QP problem via standard matlab function 'quadprog'. We recommend mosek for more efficient implementation. 
    %   QUADPROG Quadratic programming. 
    %   X = QUADPROG(H,f,A,b) attempts to solve the quadratic programming
    %   problem:
    %
    %            min 0.5*x'*H*x + f'*x   subject to:  A*x <= b
    %             x
    %
    %   X = QUADPROG(H,f,A,b,Aeq,beq) solves the problem above while
    %   additionally satisfying the equality constraints Aeq*x = beq.
    %
    %   X = QUADPROG(H,f,A,b,Aeq,beq,LB,UB) defines a set of lower and upper
    %   bounds on the design variables, X, so that the solution is in the
    %   range LB <= X <= UB. Use empty matrices for LB and UB if no bounds
    %   exist. Set LB(i) = -Inf if X(i) is unbounded below; set UB(i) = Inf if
    %   X(i) is unbounded above.
    
    var_n = ln+2;   % variable number 
    con_n = 2;      % constraint number
    
    pos_inx = u_inx(logical(tmpd(u_inx)==1));
    neg_inx = u_inx(logical(tmpd(u_inx)==-1));
    
    % Hessian matrix, i.e., H = K^{tmpd}.* (\hat{y}*\hat{y}')
    H = zeros(var_n);
    H(1:ln,1:ln) = K(l_inx,l_inx); 
    H(ln+1,1:ln) = mean(K(pos_inx,l_inx));  H(1:ln,ln+1) = H(ln+1,1:ln)';
    H(ln+2,1:ln) = mean(K(neg_inx,l_inx));  H(1:ln,ln+2) = H(ln+2,1:ln)';
    H(ln+1,ln+1) = mean(mean(K(pos_inx,pos_inx)));
    H(ln+1,ln+2) = mean(mean(K(pos_inx,neg_inx))); H(ln+2,ln+1) = H(ln+1,ln+2);
    H(ln+2,ln+2) = mean(mean(K(neg_inx,neg_inx)));      
    tr_y = [y(l_inx);1;-1];
    H = H.* (tr_y*tr_y');
    for i = 1:var_n
        H(i,i) = H(i,i) + 1e-10;
    end
    
    %linear term f
    f = zeros(var_n,1);
    f(1:ln) = -1;
        
    %Aeq
    A = zeros(con_n, var_n);
    A(1,1:var_n) = [y(l_inx)',1,-1];
    A(2,ln+1:ln+2) = [opt.cost,1];
               
    %b
    b = [0;opt.c2];
        
    %lb & ub
    lb = zeros(var_n,1);
    ub = opt.c1*ones(var_n,1);
    for kj = 1:ln
        if y(l_inx(kj)) > 0
          ub(kj) = opt.c1*opt.cost;  
        end
    end
    ub(ln+1:ln+2) = max(opt.c2/(opt.cost),opt.c2);
        
    %call quadprog   
    alpha = quadprog(H,f,[],[],A,b,lb,ub);
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2.2.2 fix SVM, refine the estimated label means
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % obtain the Support Vector
    inx = find(1e-6 < alpha(1:ln));  
 
    % calculate the rho, f(x) = w'x + rho
    pre_val = (alpha.*tr_y)'*(H.*(tr_y*tr_y'));
    rho = mean(tr_y(inx)-pre_val(inx)');
    
    % reconstruct the prdiction
    new_alpha = zeros(1,n);
    new_alpha(l_inx) = alpha(1:ln)';
    new_alpha(pos_inx) = alpha(ln+1)/length(pos_inx);
    new_alpha(neg_inx) = alpha(ln+2)/length(neg_inx);      
    pre_val = (new_alpha.*tmpd')*K + rho;
    
    % estimate the label mean
    tmptmpd = y;
    [val,ix] = sort(pre_val(u_inx),'descend');
    tmptmpd(u_inx(ix(1:opt.ep))) = 1;
    tmptmpd(u_inx(ix(opt.ep+1:un))) = -1;
    
    % check the stop condition
    if tmptmpd == tmpd
        flag = 0;
    else
        tmpd = tmptmpd; iter = iter + 1;
    end  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. train a final CS4VM with estimated label means and predict the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath('libsvm-mat-cs4vm')
opt_svm = ['-t 4 -c ' num2str(opt.c1) ' w1 ' num2str(opt.cost)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.1. preprocessing the input for final CS4VM
KK = [K(l_inx,l_inx),K(l_inx,u_inx),K(l_inx,u_inx);...
    K(u_inx,l_inx),K(u_inx,u_inx),K(u_inx,u_inx);...
    K(u_inx,l_inx),K(u_inx,u_inx),K(u_inx,u_inx)];
KK = [(1:(ln+2*un))',KK];
upperbound = zeros(ln+2*un,1);
for i = 1:ln
    if y(l_inx(i)) > 0
        upperbound(i) = opt.c1*opt.cost;
    else
        upperbound(i) = opt.c1;
    end
end
upperbound(ln+1:ln+un) = opt.c2*opt.cost;
upperbound(ln+un+1:ln+2*un) = opt.c2;

linearterm = zeros(ln+2*un,1);
posinx2 = logical(tmpd(u_inx)==1); posinx = u_inx(posinx2);
neginx2 = logical(tmpd(u_inx)==-1); neginx = u_inx(neginx2);

linearterm(1:ln) = -ones(ln,1)+y(l_inx).*(opt.c2*opt.cost*sum(K(l_inx,posinx),2)-opt.c2*sum(K(l_inx,neginx),2));
linearterm(ln+1:ln+un) = ones(un,1)- opt.c2*opt.cost*sum(K(u_inx,posinx),2)+opt.c2*sum(K(u_inx,neginx),2);
linearterm(ln+un+1:ln+2*un) = 2*ones(un,1) - linearterm(ln+1:ln+un);

zz = zeros(un,1);
zz(posinx2) = 1;
initial_point = [zeros(ln,1);opt.c2*opt.cost*zz;opt.c2*(1-zz)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.2 train CS4VM 
model = svmtrain([y(l_inx);-ones(un,1);ones(un,1)],KK,opt_svm,upperbound, linearterm,initial_point);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.3 proprocessing the output of CS4VM model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determine the dual variables alpha
tmpalpha = model.sv_coef;
alpha = zeros(ln+un,1);
alpha(l_inx) = tmpalpha(1:ln);
alpha(u_inx) = tmpalpha(ln+1:ln+un)+tmpalpha(ln+un+1:ln+2*un);

alpha(posinx) = alpha(posinx)+opt.c2*opt.cost;
alpha(neginx) = alpha(neginx)-opt.c2;

for i = 1:ln+un
    if abs(alpha(i)) < 1e-6
        alpha(i) = 0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determine the rho
tmpub = zeros(ln+un,1);
tmpub(l_inx) = upperbound(1:ln);
tmpub(posinx) = opt.c2*opt.cost;
tmpub(neginx) = opt.c2;
tmp2 = find(abs(alpha) > 0);
tmp = tmp2(abs(alpha(tmp2)) < tmpub(tmp2));
train_pred = alpha'*K;
train_pred = train_pred';

if ~isempty(tmp)
    rho = mean(tmpd(tmp)-train_pred(tmp));
else
    rho = mean(tmpd(tmp2)-train_pred(tmp2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.4 prediction 
prediction = alpha'*K_test+rho;
prediction = prediction';
label = sign(prediction);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cpu_time = cputime - tt;
