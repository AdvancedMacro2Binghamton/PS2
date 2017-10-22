clear
close all
%%%% Set up parameters
alpha = 0.35;
beta = 0.99;
delta = 0.025;
sigma = 2;
T_mat = [0.977, 1-0.977; 1 - 0.926, 0.926];
% now, we want to try different values for A_hi and A_low
T_long_run = T_mat ^ 100;
A_hi = 1.1; % A_hi = 1.0082;
A = [ A_hi, (1 - T_long_run(1,1) * A_hi) / T_long_run(1,2) ];

%%%% Set up discretized state space
k_min = 0;
k_max = 45;
num_k = 1000; % number of points in the grid for k

k = linspace(k_min, k_max, num_k);

k_mat = repmat(k', [1 num_k]); % this will be useful in a bit

%%%% Set up consumption and return function
% 1st dim(rows): k today, 2nd dim (cols): k' chosen for tomorrow
cons(:,:,1) = A(1) * k_mat .^ alpha + (1 - delta) * k_mat - k_mat'; 
cons(:,:,2) = A(2) * k_mat .^ alpha + (1 - delta) * k_mat - k_mat';

ret = cons .^ (1 - sigma) / (1 - sigma); % return function
% negative consumption is not possible -> make it irrelevant by assigning
% it very large negative utility
ret(cons < 0) = -Inf;

%%%% Iteration
dis = 1; tol = 1e-06; % tolerance for stopping 
v_guess = zeros(2, num_k);
while dis > tol
    % an alternative, more direct way to compute the value array:
    value_mat_alt = ret + beta * ...
        repmat(permute((T_mat * v_guess), [3 2 1]), [num_k 1 1]);
    
    % compute the utility value for all possible combinations of k and k':
%     value_mat(:,:,1) = ret(:,:,1) + beta * ( ...
%         T_mat(1,1) * repmat(v_guess(1,:), [num_k 1]) + ...
%         T_mat(1,2) * % finish from here!
%     
    
    % find the optimal k' for every k:
    [vfn, pol_indx] = max(value_mat_alt, [], 2);
    vfn = shiftdim(vfn,2);
    
    % what is the distance between current guess and value function
    dis = max(abs( vfn(:) - v_guess(:) ));
    
    % if distance is larger than tolerance, update current guess and
    % continue, otherwise exit the loop
    v_guess = vfn;
end
pol_indx = permute(pol_indx, [3 1 2]);

g = k(pol_indx); % policy function

plot(k,vfn)
figure
plot(k,g)
figure
plot(k, g - (1 - delta) * repmat(k, [2 1]));

%%% Simulation
% Set up simulated time series A according to T_mat
T_sim = 5000;
rng(1);
randnums = rand(T_sim, 1);
A_sim = zeros(T_sim, 1);
A_sim(1) = 1;
T_cumul = cumsum(T_mat,2);
for t = 1:T_sim-1
    if randnums(t) < T_cumul(A_sim(t),1)
        A_sim(t+1) = 1;
    else
        A_sim(t+1) = 2;
    end
end

% start with arbitrary capital stock and follow policy implied by simulated
% state
k_sim_indx = zeros(T_sim, 1);
k_sim_indx(1) = 10;
for t = 1:T_sim-1
    k_sim_indx(t+1) = pol_indx(A_sim(t), k_sim_indx(t));
end

% what is production and std dev?
y_sim = A(A_sim) .* k(k_sim_indx) .^ alpha;
y_sim(1:200) = []; % discard the first few periods
std_dev = sqrt(var(y_sim)) / mean(y_sim)

