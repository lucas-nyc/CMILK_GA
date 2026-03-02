clear
clearvars
groundtruth = readtable('groundtruth.csv'); 
fprintf('Done loading ground truth...');
load('simulation_v4.mat');

n_combinations = size(results, 1); 
n_rows = 135;
ground_truth = table2array(groundtruth); 
data_test = cell(n_combinations, 2);

for set_idx = 1:n_combinations
    fprintf('Processing combination %d\n', set_idx);
    data_test_missing = groundtruth;
    start_idx = set_idx;
    for row_idx = 1:n_rows
        combination_idx = mod(start_idx - 1 + row_idx - 1, n_combinations) + 1;
        landmark_ids = results{combination_idx, 5};
        data_test_missing{row_idx, landmark_ids} = NaN;
        scenario = results{combination_idx, 6};
    end
    data_test{set_idx,1} = data_test_missing;
    data_test{set_idx,2} = scenario;
end
fprintf('Done processing missing dataset...');

use_parallel = true;
verbose = true;
num_iterations_for_imputer = 2;
opts.PopulationSize = 40;
opts.MaxGenerations = 60;
opts.FunctionTolerance = 1e-2;

common = struct();
common.MaxEvals = opts.MaxGenerations; 
common.FunctionTolerance = opts.FunctionTolerance;
common.EarlyStopPatience = 5;
common.EarlyStopTol = common.FunctionTolerance;
common.lb = [1, 1e-6];
common.ub = [135, 1e-2];
common.UseParallel = logical(use_parallel);
common.Verbose = logical(verbose);
common.NumImputerIters = double(num_iterations_for_imputer);
common.penalty_on_error = 1e3;
common.NumWorkers = min( feature('numcores'), 6 ); % adjust max as you wish

if common.Verbose
    fprintf('Common settings: MaxEvals=%d, Patience=%d, Tol=%g, UseParallel=%d, NumWorkers=%d\n', ...
        common.MaxEvals, common.EarlyStopPatience, common.EarlyStopTol, double(common.UseParallel), common.NumWorkers);
end

if common.UseParallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            parpool('local', common.NumWorkers);
            pool = gcp();
            if common.Verbose, fprintf('Started parpool with %d workers.\n', pool.NumWorkers); end
        else
            if pool.NumWorkers ~= common.NumWorkers
                try
                    delete(pool);
                    parpool('local', common.NumWorkers);
                    pool = gcp();
                    if common.Verbose, fprintf('Restarted parpool with %d workers.\n', pool.NumWorkers); end
                catch
                    if common.Verbose, fprintf('Using existing pool with %d workers.\n', pool.NumWorkers); end
                end
            else
                if common.Verbose, fprintf('Parallel pool already running with %d workers.\n', pool.NumWorkers); end
            end
        end
    catch ME
        warning(ME.identifier, 'Could not start/adjust parpool: %s. Falling back to serial.', ME.message);
        common.UseParallel = false;
    end
end

if common.Verbose, fprintf('\n=== GA OPTIMIZATION ===\n'); end
ga_opts = optimoptions('ga', ...
    'PopulationSize', opts.PopulationSize, ...
    'MaxGenerations', opts.MaxGenerations, ...
    'FunctionTolerance', common.FunctionTolerance, ...
    'MaxStallGenerations', 5, ...
    'Display', 'iter', ...
    'UseParallel', common.UseParallel, ...
    'OutputFcn', @(options,state,flag) ga_logger(options,state,flag,common.Verbose), ...
    'PlotFcn', [] ...
    );

global GA_BEST_PARAMS GA_BEST_RMSE GA_MEAN_RMSE GA_WORST_RMSE GA_POP_LOG GA_EVAL_LOG GA_TIC GA_GEN_TIMES GA_GEN_DELTA GA_TOTAL_TIME
GA_BEST_PARAMS = []; GA_BEST_RMSE = []; GA_MEAN_RMSE = []; GA_WORST_RMSE = [];
GA_POP_LOG = struct('Generation', {}, 'Population', {}, 'Scores', {});
GA_GEN_TIMES = []; GA_GEN_DELTA = []; GA_TOTAL_TIME = NaN;
GA_TIC = tic;

fitnessFcn = @(params) timed_eval_score(params, data_test, ground_truth, common.NumImputerIters, false);

nvars = 2;
try
    [bestParams, bestScore, exitflag, output, population, scores] = ga(fitnessFcn, nvars, [], [], [], [], common.lb, common.ub, [], ga_opts);
catch ME
    warning(ME.identifier, 'GA run failed: %s', ME.message);
    bestParams = [NaN, NaN]; bestScore = NaN;
end
try
    if isempty(GA_TOTAL_TIME) || isnan(GA_TOTAL_TIME)
        GA_TOTAL_TIME = toc(GA_TIC);
    end
catch
    GA_TOTAL_TIME = NaN;
end

if ~isempty(GA_POP_LOG)
    gen_evals = arrayfun(@(s) size(s.Population,1), GA_POP_LOG);
    hist_ga.evals_per_gen = gen_evals(:)';
    hist_ga.cum_evals_per_gen = cumsum(gen_evals(:))';
    try
        ga_eval_count = sum(arrayfun(@(s) size(s.Population,1), GA_POP_LOG));
    catch
        ga_eval_count = numel(GA_BEST_RMSE) * opts.PopulationSize;
    end
else
    hist_ga.evals_per_gen = [];
    hist_ga.cum_evals_per_gen = [];
    try
        ga_eval_count = output.funccount; 
    catch
        ga_eval_count = NaN;
    end
end

best_ga.params = [round(max(1,bestParams(1))), bestParams(2)];
best_ga.rmse = bestScore;
best_ga.evals = ga_eval_count;
hist_ga.best_rmse_per_iter = GA_BEST_RMSE(:)';
hist_ga.mean_rmse_per_iter = GA_MEAN_RMSE(:)';
hist_ga.worst_rmse_per_iter = GA_WORST_RMSE(:)';
hist_ga.best_params_per_iter = GA_BEST_PARAMS;
hist_ga.gen_cumulative_time_s = GA_GEN_TIMES(:)';      
hist_ga.gen_duration_s = GA_GEN_DELTA(:)';             
ga_opts_meta = struct('PopulationSize', opts.PopulationSize, 'MaxGenerations', opts.MaxGenerations, 'UseParallel', common.UseParallel);
ga_meta.total_time_s = GA_TOTAL_TIME;
ga_meta.evals = ga_eval_count;

try
    save('ga_results.mat','best_ga','hist_ga','ga_opts_meta','ga_meta');
    if common.Verbose, fprintf('Saved ga_results.mat\n'); end
catch
    warning('Could not save ga_results.mat (no diagnostic message).');
end

if isfield(hist_ga,'best_rmse_per_iter') && ~isempty(hist_ga.best_rmse_per_iter)
    f = struct();
    f.best = safe_vec(hist_ga,'best_rmse_per_iter');
    f.mean = safe_vec(hist_ga,'mean_rmse_per_iter');
    f.worst = safe_vec(hist_ga,'worst_rmse_per_iter');
    f.cumtime = safe_vec(hist_ga,'gen_cumulative_time_s');
    f.dur = safe_vec(hist_ga,'gen_duration_s');
    f.evals = safe_vec(hist_ga,'evals_per_gen'); 
    f.cumevals = safe_vec(hist_ga,'cum_evals_per_gen'); 

    if numel(f.cumtime) == numel(f.best) + 1 && ~isempty(f.cumtime) && f.cumtime(1) == 0
        f.cumtime = f.cumtime(2:end);
        if ~isempty(f.dur) && numel(f.dur) == numel(f.cumtime)-1
        elseif numel(f.dur) == numel(f.best) + 1
            f.dur = f.dur(2:end);
        end
    end

    lens = [numel(f.best), numel(f.mean), numel(f.worst), numel(f.cumtime), numel(f.dur), numel(f.evals), numel(f.cumevals)];
    nRows = max(lens);

    fields = fieldnames(f);
    for ii = 1:numel(fields)
        v = f.(fields{ii});
        if isempty(v)
            v = nan(nRows,1);
        else
            v = v(:);                    
            if numel(v) < nRows
                v = [v; nan(nRows-numel(v),1)];
            elseif numel(v) > nRows
                v = v(1:nRows);
            end
        end
        f.(fields{ii}) = v;
    end

    Gen = (1:nRows)';
    T = table(Gen, f.best, f.mean, f.worst, f.cumtime, f.dur, f.evals, f.cumevals, ...
        'VariableNames', {'Gen','Best_RMSE','Mean_RMSE','Worst_RMSE','CumulativeTime_s','GenDuration_s','Evals_per_gen','CumulativeEvals'});
    writetable(T, 'ga_scores_per_gen.csv');
    if exist('common','var') && isfield(common,'Verbose') && common.Verbose
        fprintf('Wrote ga_scores_per_gen.csv (rows=%d)\n', nRows);
    end
end

fprintf('GA finished. best k = %d, best eps = %.8g, best RMSE = %.6g, total_time=%.2fs\n', best_ga.params(1), best_ga.params(2), best_ga.rmse, ga_meta.total_time_s);

if common.Verbose, fprintf('\n=== PSO OPTIMIZATION ===\n'); end
pso_opts = struct();
pso_opts.PopSize = opts.PopulationSize;
pso_opts.MaxIters = common.MaxEvals;
pso_opts.MaxK = common.ub(1);
pso_opts.EpsBounds = [common.lb(2), common.ub(2)];
pso_opts.Inertia = 0.7;
pso_opts.Cognitive = 1.4;
pso_opts.Social = 1.4;
pso_opts.VelClamp = 0.5;
pso_opts.Verbose = common.Verbose;
pso_opts.UseParallel = common.UseParallel;
pso_opts.EarlyStopPatience = common.EarlyStopPatience;
pso_opts.EarlyStopTol = common.FunctionTolerance;

try
    tic_ps = tic;
    [best_pso, hist_pso] = pso_optimize_cmilk(data_test, ground_truth, common.NumImputerIters, pso_opts);
    t_ps = toc(tic_ps);
    if common.Verbose
        fprintf('PSO finished: best k=%d, eps=%.8g, RMSE=%.6g, evals=%d, time=%.2fs\n', ...
            best_pso.params(1), best_pso.params(2), best_pso.rmse, best_pso.evals, t_ps);
    end
catch ME
    warning(ME.identifier,  'PSO failed: %s', ME.message);
    best_pso = struct('params',[NaN,NaN],'rmse',NaN,'evals',NaN); hist_pso = struct();
end
try
    save('pso_results.mat', 'best_pso', 'hist_pso', 'pso_opts');
    if isfield(hist_pso,'best_rmse_per_iter')
        nIter = numel(hist_pso.best_rmse_per_iter);
        T = table((1:nIter)', hist_pso.best_rmse_per_iter(:), hist_pso.mean_score_per_iter(:), hist_pso.worst_score_per_iter(:), ...
            hist_pso.cum_time(:), hist_pso.time_per_iter(:), hist_pso.evals_per_iter(:), hist_pso.cum_evals(:), ...
            'VariableNames', {'Iter','Best_RMSE','Mean_RMSE','Worst_RMSE','CumulativeTime_s','GenDuration_s','Evals_per_iter','CumulativeEvals'});
        writetable(T, 'pso_scores_per_iter.csv');
    end
    if common.Verbose, fprintf('Saved pso_results.mat and pso_scores_per_iter.csv (if available)\n'); end
catch ME
    warning(ME.identifier,  'Could not save PSO outputs: %s', ME.message);
end

if common.Verbose, fprintf('\n=== DE OPTIMIZATION ===\n'); end
de_opts = struct();
de_opts.PopSize = opts.PopulationSize;
de_opts.MaxIters = common.MaxEvals;
de_opts.MaxK = common.ub(1);
de_opts.EpsBounds = [common.lb(2), common.ub(2)];
de_opts.F = 0.8;
de_opts.CR = 0.9;
de_opts.Verbose = common.Verbose;
de_opts.UseParallel = common.UseParallel;
de_opts.EarlyStopPatience = common.EarlyStopPatience;
de_opts.EarlyStopTol = common.FunctionTolerance;

try
    tic_de = tic;
    [best_de, hist_de] = de_optimize_cmilk(data_test, ground_truth, common.NumImputerIters, de_opts);
    t_de = toc(tic_de);
    if common.Verbose
        fprintf('DE finished: best k=%d, eps=%.8g, RMSE=%.6g, evals=%d, time=%.2fs\n', ...
            best_de.params(1), best_de.params(2), best_de.rmse, best_de.evals, t_de);
    end
catch ME
    warning(ME.identifier,  'DE failed: %s', ME.message);
    best_de = struct('params',[NaN,NaN],'rmse',NaN,'evals',NaN); hist_de = struct();
end

try
    save('de_results.mat', 'best_de', 'hist_de', 'de_opts');
    if isfield(hist_de,'best_rmse_per_iter')
        nIter = numel(hist_de.best_rmse_per_iter);
        T = table((1:nIter)', hist_de.best_rmse_per_iter(:), hist_de.mean_score_per_iter(:), hist_de.worst_score_per_iter(:), ...
            hist_de.cum_time(:), hist_de.time_per_iter(:), hist_de.evals_per_iter(:), hist_de.cum_evals(:), ...
            'VariableNames', {'Iter','Best_RMSE','Mean_RMSE','Worst_RMSE','CumulativeTime_s','GenDuration_s','Evals_per_iter','CumulativeEvals'});
        writetable(T, 'de_scores_per_iter.csv');
    end
    if common.Verbose, fprintf('Saved de_results.mat and de_scores_per_iter.csv (if available)\n'); end
catch ME
    warning(ME.identifier, 'Could not save DE outputs: %s', ME.message);
end

if common.Verbose, fprintf('\n=== GWO OPTIMIZATION ===\n'); end
gwo_opts = struct();
gwo_opts.PopSize = opts.PopulationSize;
gwo_opts.MaxIters = common.MaxEvals;
gwo_opts.UseParallel = common.UseParallel;
gwo_opts.MaxK = common.ub(1);
gwo_opts.EpsBounds = [common.lb(2), common.ub(2)];
gwo_opts.Verbose = common.Verbose;
gwo_opts.EarlyStopPatience = common.EarlyStopPatience;
gwo_opts.EarlyStopTol = common.FunctionTolerance;

try
    tic_gwo = tic;
    [best_gwo, hist_gwo] = gwo_optimize_cmilk(data_test, ground_truth, common.NumImputerIters, gwo_opts);
    t_gwo = toc(tic_gwo);
    if common.Verbose
        fprintf('GWO finished: best k=%d, eps=%.8g, RMSE=%.6g, evals=%d, time=%.2fs\n', ...
            best_gwo.params(1), best_gwo.params(2), best_gwo.rmse, best_gwo.evals, t_gwo);
    end
catch ME
    warning(ME.identifier, 'GWO failed: %s', ME.message);
    best_gwo = struct('params',[NaN,NaN],'rmse',NaN,'evals',NaN); hist_gwo = struct();
end

try
    save('gwo_results.mat', 'best_gwo', 'hist_gwo', 'gwo_opts');
    if isfield(hist_gwo,'best_rmse_per_iter')
        nIter = numel(hist_gwo.best_rmse_per_iter);
        T = table((1:nIter)', hist_gwo.best_rmse_per_iter(:), hist_gwo.mean_score_per_iter(:), hist_gwo.worst_score_per_iter(:), ...
            hist_gwo.cum_time(:), hist_gwo.time_per_iter(:), hist_gwo.evals_per_iter(:), hist_gwo.cum_evals(:), ...
            'VariableNames', {'Iter','Best_RMSE','Mean_RMSE','Worst_RMSE','CumulativeTime_s','GenDuration_s','Evals_per_iter','CumulativeEvals'});
        writetable(T, 'gwo_scores_per_iter.csv');
    end
    if common.Verbose, fprintf('Saved gwo_results.mat and gwo_scores_per_iter.csv (if available)\n'); end
catch ME
    warning(ME.identifier, 'Could not save GWO outputs: %s', ME.message);
end

try
    Methods = {}; Bk = []; Be = []; Br = []; Ev = [];
    if exist('best_ga','var'),   Methods{end+1}='GA';    Bk(end+1)=best_ga.params(1);   Be(end+1)=best_ga.params(2);   Br(end+1)=best_ga.rmse;   Ev(end+1)=best_ga.evals; end
    if exist('best_pso','var'),  Methods{end+1}='PSO';   Bk(end+1)=best_pso.params(1);  Be(end+1)=best_pso.params(2);  Br(end+1)=best_pso.rmse;  Ev(end+1)=best_pso.evals; end
    if exist('best_de','var'),   Methods{end+1}='DE';    Bk(end+1)=best_de.params(1);   Be(end+1)=best_de.params(2);   Br(end+1)=best_de.rmse;   Ev(end+1)=best_de.evals; end
    if exist('best_gwo','var'),  Methods{end+1}='GWO';   Bk(end+1)=best_gwo.params(1);  Be(end+1)=best_gwo.params(2);  Br(end+1)=best_gwo.rmse;  Ev(end+1)=best_gwo.evals; end

    Tsum = table(Methods', Bk', Be', Br', Ev', 'VariableNames', {'Method','Best_k','Best_eps','Best_RMSE','Evaluations'});
    writetable(Tsum, 'optimizers_summary_all_extended.csv');
    if common.Verbose, fprintf('Wrote optimizers_summary_all_extended.csv\n'); end
catch ME
    warning(ME.identifier, 'Could not write extended summary CSV: %s', ME.message);
end

fprintf('\nAll optimization runs complete.\n');
%% --------------- Plotting -------------------
clf;
optimizers = {'pso','de','gwo','ga'};
labels = {'PSO','DE','GWO','GA'};
mat_files = {'pso_results.mat','de_results.mat','gwo_results.mat','ga_results.mat'};
csv_files = {'pso_best_rmse_per_iter.csv','de_best_rmse_per_gen.csv','gwo_best_rmse_per_iter.csv','ga_best_rmse_per_gen.csv'};
best_struct_names = {'best_pso','best_de','best_gwo','best_ga'};
hist_vecs = cell(size(optimizers));
best_structs = repmat(struct('params',[NaN,NaN],'rmse',NaN,'evals',NaN), numel(optimizers), 1);

for i = 1:numel(optimizers)
    hist_vecs{i} = load_history_vector(mat_files{i}, csv_files{i});
    if exist(mat_files{i}, 'file')
        try
            S = load(mat_files{i});
            fn = best_struct_names{i};
            if isfield(S, fn)
                best_structs(i) = S.(fn);
            elseif isfield(S,'best_ga') && strcmp(optimizers{i},'ga')
                best_structs(i) = S.best_ga;
            elseif isfield(S,'best_pso') && strcmp(optimizers{i},'pso')
                best_structs(i) = S.best_pso;
            elseif isfield(S,'best_de') && strcmp(optimizers{i},'de')
                best_structs(i) = S.best_de;
            elseif isfield(S,'best_gwo') && strcmp(optimizers{i},'gwo')
                best_structs(i) = S.best_gwo;
            end
        catch
        end
    end
end
Method = labels(:);
Best_k = NaN(numel(Method),1);
Best_eps = NaN(numel(Method),1);
Best_RMSE = NaN(numel(Method),1);
Evaluations = NaN(numel(Method),1);

for i = 1:numel(Method)
    b = best_structs(i);
    if isstruct(b) && isfield(b,'params')
        Best_k(i) = b.params(1);
        Best_eps(i) = b.params(2);
    end
    if isstruct(b) && isfield(b,'rmse')
        Best_RMSE(i) = b.rmse;
    end
    if isstruct(b) && isfield(b,'evals')
        Evaluations(i) = b.evals;
    end
end

T = table(Method, Best_k, Best_eps, Best_RMSE, Evaluations);
writetable(T, 'optimizers_summary_all.csv');
fprintf('Wrote optimizers_summary_all.csv\n');

maxIter = 18;
verbose = true;

if ~exist('hist_pso','var') && exist('pso_results.mat','file')
    tmp = load('pso_results.mat'); if isfield(tmp,'hist_pso'), hist_pso = tmp.hist_pso; end; end
if ~exist('hist_de','var') && exist('de_results.mat','file')
    tmp = load('de_results.mat'); if isfield(tmp,'hist_de'), hist_de = tmp.hist_de; end; end
if ~exist('hist_gwo','var') && exist('gwo_results.mat','file')
    tmp = load('gwo_results.mat'); if isfield(tmp,'hist_gwo'), hist_gwo = tmp.hist_gwo; end; end
if ~exist('hist_ga','var') && exist('ga_results.mat','file')
    tmp = load('ga_results.mat'); if isfield(tmp,'hist_ga'), hist_ga = tmp.hist_ga; end; end

meanCandidates = {'mean_score_per_iter','mean_rmse_per_iter','mean_score_per_gen','mean_rmse_per_gen','mean_score','mean_rmse'};
bestParamsCandidates = {'best_params_per_iter','best_params','global_best_params','global_best','best_params_per_gen'};
bestScalarCandidates = {'best_rmse_per_iter','best_rmse_per_gen','best_score_per_iter','best_score_per_gen','best_rmse','best_score'};
timeCandidates = {'time_per_iter','time_per_gen','GenDuration_s','gen_duration_s','GenDuration','gen_duration','cum_time','CumulativeTime_s'};

pso_mean = extract_vec_safe(hist_pso, meanCandidates);
de_mean  = extract_vec_safe(hist_de,  meanCandidates);
gwo_mean = extract_vec_safe(hist_gwo, meanCandidates);
ga_mean  = extract_vec_safe(hist_ga,  meanCandidates);

if isempty(pso_mean), pso_mean = extract_vec_safe(hist_pso, bestScalarCandidates); end
if isempty(de_mean),  de_mean  = extract_vec_safe(hist_de,  bestScalarCandidates); end
if isempty(gwo_mean), gwo_mean = extract_vec_safe(hist_gwo, bestScalarCandidates); end
if isempty(ga_mean),  ga_mean  = extract_vec_safe(hist_ga,  bestScalarCandidates); end

ga_best = extract_vec_safe(hist_ga, bestScalarCandidates);

pso_params = extract_mat2_safe(hist_pso, bestParamsCandidates);
de_params  = extract_mat2_safe(hist_de,  bestParamsCandidates);
gwo_params = extract_mat2_safe(hist_gwo, bestParamsCandidates);
ga_params  = extract_mat2_safe(hist_ga,  bestParamsCandidates);

pso_time = extract_vec_safe(hist_pso, timeCandidates);
de_time  = extract_vec_safe(hist_de,  timeCandidates);
gwo_time = extract_vec_safe(hist_gwo, timeCandidates);

ga_gen_duration = [];
if isstruct(hist_ga) && isfield(hist_ga,'gen_duration_s') && isnumeric(hist_ga.gen_duration_s)
    ga_gen_duration = hist_ga.gen_duration_s(:);
elseif isstruct(hist_ga) && isfield(hist_ga,'GenDuration_s') && isnumeric(hist_ga.GenDuration_s)
    ga_gen_duration = hist_ga.GenDuration_s(:);
else
    ga_gen_duration = extract_vec_safe(hist_ga, timeCandidates);
end

availableLens = [numel(pso_mean), numel(de_mean), numel(gwo_mean), numel(ga_mean), ...
    size(pso_params,1), size(de_params,1), size(gwo_params,1), size(ga_params,1), ...
    numel(pso_time), numel(de_time), numel(gwo_time), numel(ga_gen_duration), numel(ga_best)];
nAvail = max([availableLens, 0]);
nIter = min(maxIter, max(1, nAvail)); 
pso_plot = pad_or_nan(pso_mean, nIter);
de_plot  = pad_or_nan(de_mean,  nIter);
gwo_plot = pad_or_nan(gwo_mean, nIter);
ga_plot  = pad_or_nan(ga_mean,  nIter);
ga_best_plot = pad_or_nan(ga_best, nIter);

pso_mat = pad_mat2(pso_params, nIter);
de_mat  = pad_mat2(de_params, nIter);
gwo_mat = pad_mat2(gwo_params, nIter);
ga_mat  = pad_mat2(ga_params, nIter);

pso_time_p = pad_or_nan(pso_time, nIter);
de_time_p  = pad_or_nan(de_time, nIter);
gwo_time_p = pad_or_nan(gwo_time, nIter);
ga_gen_dur  = pad_or_nan(ga_gen_duration, nIter);

Iter = (1:nIter)';
CombinedTable = table(Iter, ...
    pso_plot, pso_mat(:,1), pso_mat(:,2), pso_time_p, ...
    de_plot,  de_mat(:,1),  de_mat(:,2),  de_time_p, ...
    gwo_plot, gwo_mat(:,1), gwo_mat(:,2), gwo_time_p, ...
    ga_plot,  ga_mat(:,1),  ga_mat(:,2),  ga_gen_dur, ...
    'VariableNames', {'Iter', ...
        'PSO_mean_rmse','PSO_best_k','PSO_best_eps','PSO_time_s', ...
        'DE_mean_rmse','DE_best_k','DE_best_eps','DE_time_s', ...
        'GWO_mean_rmse','GWO_best_k','GWO_best_eps','GWO_time_s', ...
        'GA_mean_rmse','GA_best_k','GA_best_eps','GA_genDuration_s' });

writetable(CombinedTable,'optimizers_combined_iter_table.csv');
save('optimizers_combined_iter_table.mat','CombinedTable');
if verbose
    fprintf('Saved optimizers_combined_iter_table.csv and .mat (nIter=%d)\n', nIter);
    disp(CombinedTable);
end

x = (1:nIter)';
figure('Units','normalized','Position',[0.15 0.15 0.7 0.6]); clf; hold on; grid on;
if any(isfinite(pso_plot)), plot(x, pso_plot, '-o', 'DisplayName','PSO mean', 'LineWidth',1.4); end
if any(isfinite(de_plot)),  plot(x, de_plot,  '-s', 'DisplayName','DE mean',  'LineWidth',1.4); end
if any(isfinite(gwo_plot)), plot(x, gwo_plot, '-^', 'DisplayName','GWO mean', 'LineWidth',1.4); end
if any(isfinite(ga_plot)),  plot(x, ga_plot,  '-d', 'DisplayName','GA mean',  'LineWidth',1.4); end

if any(isfinite(ga_best_plot))
    plot(x, ga_best_plot, '--k', 'DisplayName','GA best (single)', 'LineWidth',2.0);
end

xlim([1 maxIter]); xlabel('Iteration'); ylabel('RMSE');
title(sprintf('Mean RMSE convergence (first %d iterations) — GA best overlaid', maxIter));
legend('Location','best');

allvals = [pso_plot; de_plot; gwo_plot; ga_plot; ga_best_plot];
if any(isfinite(allvals))
    finiteVals = allvals(isfinite(allvals));
    ymi = min(finiteVals); yma = max(finiteVals);
    yrange = yma - ymi;
    if yrange == 0
        ylim([ymi - 0.1*abs(ymi+eps), yma + 0.1*abs(yma+eps)]);
    else
        ylim([ymi - 0.12*yrange, yma + 0.12*yrange]);
    end
end

if any(isfinite(ga_best_plot))
    idx = find(isfinite(ga_best_plot),1,'last');
    text(x(idx), ga_best_plot(idx), sprintf('  best=%.4g', ga_best_plot(idx)), 'FontSize',9, 'VerticalAlignment','bottom');
end

saveas(gcf,'optimizers_mean_convergence_combined.png');
if verbose, fprintf('Saved optimizers_mean_convergence_combined.png\n'); end

%% --------------------- Ablation Study ---------------------
k_best = 19;
eps_best = 0.0062039165;
Kmax = 135;
ks = (1:Kmax)';
Neps = 40;
eps_vals = logspace(-6, -2, Neps)';

n_sets = n_combinations;

RMSE_by_k = nan(Kmax, n_sets);
MAE_by_k  = nan(Kmax, n_sets);


fprintf('Running k sweep (eps fixed at %g)\n', eps_best);
for k_val = ks'
    fprintf('k = %d\n', k_val);
    for set_idx = 1:n_sets
        fprintf('Processing missing dataset %d/%d (k=%d)\n', set_idx, n_sets, k_val);
        modified_dataset = table2array(data_test{set_idx,1});
        missing_mask = isnan(modified_dataset);
        if ~any(missing_mask(:))
            RMSE_by_k(k_val, set_idx) = NaN;
            MAE_by_k(k_val, set_idx) = NaN;
            continue
        end
        imputed_dataset = run_cmilk_single(modified_dataset, k_val, eps_best, num_cols, num_rows, threshold_ga);
        imputed_vals = imputed_dataset(missing_mask);
        true_vals = data(missing_mask);
        valid = ~isnan(imputed_vals) & ~isnan(true_vals);
        if any(valid)
            errs = imputed_vals(valid) - true_vals(valid);
            RMSE_by_k(k_val, set_idx) = sqrt(mean(errs.^2));
            MAE_by_k(k_val, set_idx)  = mean(abs(errs));
        else
            RMSE_by_k(k_val, set_idx) = NaN;
            MAE_by_k(k_val, set_idx)  = NaN;
        end
    end
end

mean_RMSE = mean(RMSE_by_k, 2, 'omitnan');
mean_MAE  = mean(MAE_by_k, 2, 'omitnan');

RMSE_by_eps = nan(Neps, n_sets);
MAE_by_eps  = nan(Neps, n_sets);

fprintf('Running eps sweep (k fixed at %d)\n', k_best);
for i = 1:Neps
    eps_val = eps_vals(i);
    fprintf('eps = %g (%d/%d)\n', eps_val, i, Neps);
    for set_idx = 1:n_sets
        fprintf('Processing missing dataset %d/%d (eps=%d)\n', set_idx, n_sets, eps_val);
        modified_dataset = table2array(data_test{set_idx,1});
        missing_mask = isnan(modified_dataset);
        if ~any(missing_mask(:))
            RMSE_by_eps(i, set_idx) = NaN;
            MAE_by_eps(i, set_idx) = NaN;
            continue
        end
        imputed_dataset = run_cmilk_single(modified_dataset, k_best, eps_val, num_cols, num_rows, threshold_ga);
        imputed_vals = imputed_dataset(missing_mask);
        true_vals = data(missing_mask);
        valid = ~isnan(imputed_vals) & ~isnan(true_vals);
        if any(valid)
            errs = imputed_vals(valid) - true_vals(valid);
            RMSE_by_eps(i, set_idx) = sqrt(mean(errs.^2));
            MAE_by_eps(i, set_idx)  = mean(abs(errs));
        else
            RMSE_by_eps(i, set_idx) = NaN;
            MAE_by_eps(i, set_idx)  = NaN;
        end
    end
end

fprintf('Plotting k & eps vs RMSE & MAE...');

mean_RMSE_eps = mean(RMSE_by_eps, 2, 'omitnan');
mean_MAE_eps  = mean(MAE_by_eps, 2, 'omitnan'); 

% ------------------ Plotting ----------------
figure(1); clf; hold on;
grid on; 
ax = gca;
ax.GridLineWidth = 2;
ax.LineWidth = 2;
ax.FontSize = 20;
semilogx(eps_vals, mean_RMSE_eps, '-o', 'LineWidth', 1.8, 'MarkerSize', 3, 'Color', [0.1 0.4 0.8]);
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 54);
ylabel('$\overline{\mathrm{RMSE}}$', 'Interpreter', 'latex', 'FontSize', 54);
title(['\textbf{$\overline{\mathrm{RMSE}}$ vs $\varepsilon$ (k=', num2str(k_best), ')}'], ...
      'Interpreter', 'latex', 'FontSize', 60);
set(gcf, 'Position', get(0, 'ScreenSize'));
drawnow;
exportgraphics(gcf, 'mean_RMSE_vs_eps_v2.png', 'Resolution', 300);

figure(2); clf; hold on;
grid on
ax = gca;
ax.GridLineWidth = 2;
ax.LineWidth = 2;
ax.FontSize = 20;
semilogx(eps_vals, mean_MAE_eps, '-s', 'LineWidth', 1.8, 'MarkerSize', 3, 'Color', [0.85 0.33 0.1]);
xlabel('$\varepsilon$', 'Interpreter', 'latex', 'FontSize', 54);
ylabel('$\overline{\mathrm{MAE}}$', 'Interpreter', 'latex', 'FontSize', 54);
title(['\textbf{$\overline{\mathrm{MAE}}$ vs $\varepsilon$ (k=', num2str(k_best), ')}'], ...
      'Interpreter', 'latex', 'FontSize', 60);
set(gcf, 'Position', get(0, 'ScreenSize'));
drawnow;
exportgraphics(gcf, 'mean_MAE_vs_eps_v2.png', 'Resolution', 300);

figure(3); clf; hold on; grid on;
ax = gca;
ax.GridLineWidth = 2;
ax.LineWidth = 2;
ax.FontSize = 20;
plot(ks, mean_RMSE, '-o', 'LineWidth', 1.8, 'MarkerSize', 3, 'Color', [0.1 0.4 0.8]);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$\overline{\mathrm{RMSE}}$', 'Interpreter', 'latex', 'FontSize', 50);
title(['\textbf{$\overline{\mathrm{RMSE}}$ vs $k$ ($\varepsilon$=', num2str(eps_best), ')}'], ...
      'Interpreter', 'latex', 'FontSize', 60);
set(gcf, 'Position', get(0, 'ScreenSize'));
drawnow;
exportgraphics(gcf, 'mean_RMSE_vs_k_v2.png', 'Resolution', 300);

figure(4); clf; hold on; grid on;
ax = gca;
ax.GridLineWidth = 2;
ax.LineWidth = 2;
ax.FontSize = 20;
plot(ks, mean_MAE, '-o', 'LineWidth', 1.8, 'MarkerSize', 3, 'Color', [0.85 0.33 0.1]);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$\overline{\mathrm{MAE}}$', 'Interpreter', 'latex', 'FontSize', 50);
title(['\textbf{$\overline{\mathrm{MAE}}$ vs $k$ ($\varepsilon$=', num2str(eps_best), ')}'], ...
      'Interpreter', 'latex', 'FontSize', 60);
set(gcf, 'Position', get(0, 'ScreenSize'));
drawnow;
exportgraphics(gcf, 'mean_MAE_vs_k_v2.png', 'Resolution', 300);

fprintf('Done abalation study');
%% -------------------- PSO --------------------
function [best, history] = pso_optimize_cmilk(data_test_missing_sets, ground_truth, num_iterations, opts)

    if nargin < 4, opts = struct(); end
    PopSize = getopt(opts,'PopSize',30);
    GMaxIters = getopt(opts,'MaxIters',100);
    MaxK = getopt(opts,'MaxK',50);
    EpsBounds = getopt(opts,'EpsBounds',[1e-8,1e-1]);
    w = getopt(opts,'Inertia',0.7);
    c1 = getopt(opts,'Cognitive',1.4);
    c2 = getopt(opts,'Social',1.4);
    VelClampFrac = getopt(opts,'VelClamp',0.5);
    verbose = getopt(opts,'Verbose',false);
    use_parallel = getopt(opts,'UseParallel', false);
    patience = getopt(opts,'EarlyStopPatience', 10);
    tol = getopt(opts,'EarlyStopTol', 1e-6);

    if use_parallel
        try p = gcp('nocreate'); if isempty(p), use_parallel = false; end; catch, use_parallel = false; end
    end

    k_lb = 1; k_ub = MaxK;
    eps_lb = EpsBounds(1); eps_ub = EpsBounds(2);

    rng('shuffle');
    X = zeros(PopSize,2);
    X(:,1) = k_lb + (k_ub-k_lb).*rand(PopSize,1);
    X(:,2) = eps_lb + (eps_ub-eps_lb).*rand(PopSize,1);
    v = zeros(PopSize,2);
    v(:,1) = -VelClampFrac*(k_ub-k_lb) + 2*VelClampFrac*(k_ub-k_lb).*rand(PopSize,1);
    v(:,2) = -VelClampFrac*(eps_ub-eps_lb) + 2*VelClampFrac*(eps_ub-eps_lb).*rand(PopSize,1);

    pbest = X;
    pbest_score = inf(PopSize,1);
    gbest = [];
    gbest_score = inf;
    history.best_rmse_per_iter = nan(GMaxIters,1);
    history.mean_score_per_iter = nan(GMaxIters,1);
    history.worst_score_per_iter = nan(GMaxIters,1);
    history.time_per_iter = nan(GMaxIters,1);       
    history.cum_time = nan(GMaxIters,1);
    history.global_best_params = nan(GMaxIters,2);
    history.evals_per_iter = nan(GMaxIters,1);
    history.time_per_eval_mean = nan(GMaxIters,1);
    history.time_per_eval_sum = nan(GMaxIters,1);
    history.cum_evals = nan(GMaxIters,1);

    eval_count = 0;
    no_improve = 0;
    cumtime = 0;

    for iter = 1:GMaxIters
        if verbose, fprintf('PSO: iter %d/%d\n', iter, GMaxIters); end
        t0 = tic;
        scores = inf(PopSize,1);
        eval_times = nan(PopSize,1);
        
        if use_parallel
            parfor i = 1:PopSize
                params_cont = X(i,:);
                params_eval = [round(max(1, params_cont(1))), params_cont(2)];
                params_eval(1) = min(max(params_eval(1), k_lb), k_ub);
                params_eval(2) = min(max(params_eval(2), eps_lb), eps_ub);
                [s, dt] = timed_eval(params_eval, data_test_missing_sets, ground_truth, num_iterations, false);
                scores(i) = s;
                eval_times(i) = dt;
            end
        else
            for i = 1:PopSize
                params_cont = X(i,:);
                params_eval = [round(max(1, params_cont(1))), params_cont(2)];
                params_eval(1) = min(max(params_eval(1), k_lb), k_ub);
                params_eval(2) = min(max(params_eval(2), eps_lb), eps_ub);
                [s, dt] = timed_eval(params_eval, data_test_missing_sets, ground_truth, num_iterations, false);
                scores(i) = s;
                eval_times(i) = dt;
            end
        end
        
        iter_time = toc(t0);
        cumtime = cumtime + iter_time;
        eval_count = eval_count + PopSize;
        
        history.evals_per_iter(iter) = PopSize;
        history.time_per_eval_mean(iter) = mean(eval_times(isfinite(eval_times)));
        history.time_per_eval_sum(iter) = sum(eval_times(isfinite(eval_times)));
        history.cum_evals(iter) = eval_count;

        valid_idx = isfinite(scores);
        if any(valid_idx)
            mean_score = mean(scores(valid_idx));
            worst_score = max(scores(valid_idx));
            cur_best_gen = min(scores(valid_idx));
        else
            mean_score = NaN; worst_score = NaN; cur_best_gen = NaN;
        end

        improved = scores < pbest_score;
        pbest_score(improved) = scores(improved);
        pbest(improved,:) = X(improved,:);

        [minscore, minidx] = min(pbest_score);
        improved_global = false;
        if minscore < gbest_score - tol
            gbest_score = minscore;
            gbest = pbest(minidx,:);
            improved_global = true;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        % store history
        history.best_rmse_per_iter(iter) = gbest_score;
        history.mean_score_per_iter(iter) = mean_score;
        history.worst_score_per_iter(iter) = worst_score;
        history.time_per_iter(iter) = iter_time;
        history.cum_time(iter) = cumtime;
        history.global_best_params(iter,:) = gbest;
        for i = 1:PopSize
            r1 = rand(1,2);
            r2 = rand(1,2);
            v(i,:) = w.*v(i,:) + c1.*r1.*(pbest(i,:) - X(i,:)) + c2.*r2.*(gbest - X(i,:));
        end
        v(:,1) = max(min(v(:,1), VelClampFrac*(k_ub-k_lb)), -VelClampFrac*(k_ub-k_lb));
        v(:,2) = max(min(v(:,2), VelClampFrac*(eps_ub-eps_lb)), -VelClampFrac*(eps_ub-eps_lb));

        X = X + v;
        X(:,1) = min(max(X(:,1), k_lb), k_ub);
        X(:,2) = min(max(X(:,2), eps_lb), eps_ub);

        if verbose
            fprintf('  iter %d best_so_far=%.6g (k~%.3g, eps=%.6g) mean=%.6g worst=%.6g dt=%.2fs\n', ...
                iter, gbest_score, gbest(1), gbest(2), mean_score, worst_score, iter_time);
        end

        if no_improve >= patience
            if verbose, fprintf('PSO early stopping at iter %d (no improvement for %d iters)\n', iter, no_improve); end
            history = trim_history(history, iter);
            break;
        end
    end

    best.params = [round(gbest(1)), gbest(2)];
    best.rmse = gbest_score;
    best.evals = eval_count;
end

%% -------------------- DE -----------------------
function [best, history] = de_optimize_cmilk(data_test_missing_sets, ground_truth, num_iterations, opts)
    if nargin < 4, opts = struct(); end
    NP = getopt(opts,'PopSize',40);
    GMaxIters = getopt(opts,'MaxIters',100);
    MaxK = getopt(opts,'MaxK',50);
    EpsBounds = getopt(opts,'EpsBounds',[1e-8,1e-1]);
    F = getopt(opts,'F',0.8);
    CR = getopt(opts,'CR',0.9);
    verbose = getopt(opts,'Verbose',false);
    use_parallel = getopt(opts,'UseParallel', false);
    patience = getopt(opts,'EarlyStopPatience', 10);
    tol = getopt(opts,'EarlyStopTol', 1e-6);
    history.evals_per_iter = nan(GMaxIters,1);
    history.time_per_eval_mean = nan(GMaxIters,1);
    history.time_per_eval_sum = nan(GMaxIters,1);

    if use_parallel
        try p =gcp('nocreate');
            if isempty(p)
                use_parallel = false;
            end
        catch
            use_parallel = false;
        end
    end


    k_lb = 1; k_ub = MaxK;
    eps_lb = EpsBounds(1); eps_ub = EpsBounds(2);

    rng('shuffle');
    pop = zeros(NP,2);
    pop(:,1) = k_lb + (k_ub-k_lb).*rand(NP,1);
    pop(:,2) = eps_lb + (eps_ub-eps_lb).*rand(NP,1);

    t_init = tic;
    pop_scores = inf(NP,1);
    pop_eval_times = nan(NP,1);
    if use_parallel
        parfor i = 1:NP
            params_eval = [round(max(1, pop(i,1))), pop(i,2)];
            params_eval(1) = min(max(params_eval(1), k_lb), k_ub);
            params_eval(2) = min(max(params_eval(2), eps_lb), eps_ub);
            [s, dt] = timed_eval(params_eval, data_test_missing_sets, ground_truth, num_iterations, false);
            pop_scores(i) = s;
            pop_eval_times(i) = dt;
        end
    else
        for i = 1:NP
            params_eval = [round(max(1, pop(i,1))), pop(i,2)];
            params_eval(1) = min(max(params_eval(1), k_lb), k_ub);
            params_eval(2) = min(max(params_eval(2), eps_lb), eps_ub);
            [s, dt] = timed_eval(params_eval, data_test_missing_sets, ground_truth, num_iterations, false);
            pop_scores(i) = s;
            pop_eval_times(i) = dt;
        end
    end
    init_eval_time = toc(t_init);   
    eval_count = NP;
    history.init_eval_time = init_eval_time;
    history.init_evals = NP;
    history.init_time_per_eval_mean = mean(pop_eval_times(isfinite(pop_eval_times)));
    history.init_time_per_eval_sum = sum(pop_eval_times(isfinite(pop_eval_times)));
    history.cum_evals = nan(GMaxIters,1);
    cumtime = init_eval_time;  
    no_improve = 0;
    [best_score, idx] = min(pop_scores);
    best_ind = pop(idx,:);

    for gen = 1:GMaxIters
        if verbose, fprintf('DE: gen %d/%d best=%.6g\n', gen, GMaxIters, best_score); end
        t0 = tic;

        mutants = zeros(NP,2); trials = zeros(NP,2);
        for i = 1:NP
            idxs = randperm(NP, 3);
            while any(idxs == i), idxs = randperm(NP,3); end
            a=idxs(1); b=idxs(2); c=idxs(3);
            mutant = pop(a,:) + F*(pop(b,:) - pop(c,:));
            mutant(1) = min(max(mutant(1), k_lb), k_ub);
            mutant(2) = min(max(mutant(2), eps_lb), eps_ub);
            mutants(i,:) = mutant;
            trial = pop(i,:);
            jrand = randi(2);
            for j=1:2
                if rand <= CR || j == jrand
                    trial(j) = mutant(j);
                end
            end
            trials(i,:) = trial;
        end

        trial_scores = inf(NP,1);
        trial_eval_times = nan(NP,1);
        if use_parallel
            parfor i = 1:NP
                trial_eval = [round(max(1,trials(i,1))), trials(i,2)];
                trial_eval(1)=min(max(trial_eval(1),k_lb),k_ub);
                trial_eval(2)=min(max(trial_eval(2),eps_lb),eps_ub);
                [s, dt] = timed_eval(trial_eval, data_test_missing_sets, ground_truth, num_iterations, false);
                trial_scores(i) = s;
                trial_eval_times(i) = dt;
            end
        else
            for i=1:NP
                trial_eval = [round(max(1,trials(i,1))), trials(i,2)];
                trial_eval(1)=min(max(trial_eval(1),k_lb),k_ub);
                trial_eval(2)=min(max(trial_eval(2),eps_lb),eps_ub);
                [s, dt] = timed_eval(trial_eval, data_test_missing_sets, ground_truth, num_iterations, false);
                trial_scores(i) = s;
                trial_eval_times(i) = dt;
            end
        end
        
        eval_count = eval_count + NP;
        
        history.evals_per_iter(gen) = NP;
        history.time_per_eval_mean(gen) = mean(trial_eval_times(isfinite(trial_eval_times)));
        history.time_per_eval_sum(gen) = sum(trial_eval_times(isfinite(trial_eval_times)));
        history.cum_evals(gen) = eval_count;

        for i = 1:NP
            if trial_scores(i) <= pop_scores(i)
                pop(i,:) = trials(i,:);
                pop_scores(i) = trial_scores(i);
                if trial_scores(i) < best_score - tol
                    best_score = trial_scores(i);
                    best_ind = trials(i,:);
                    no_improve = 0;
                end
            end
        end

        valid_idx = isfinite(pop_scores);
        if any(valid_idx)
            mean_score = mean(pop_scores(valid_idx));
            worst_score = max(pop_scores(valid_idx));
        else
            mean_score = NaN; worst_score = NaN;
        end

        iter_time = toc(t0);
        cumtime = cumtime + iter_time;

        history.best_rmse_per_iter(gen) = best_score;
        history.mean_score_per_iter(gen) = mean_score;
        history.worst_score_per_iter(gen) = worst_score;
        history.time_per_iter(gen) = iter_time;
        history.cum_time(gen) = cumtime;
        history.best_params_per_iter(gen,:) = best_ind;

        if verbose
            fprintf('  gen %d best=%.6g mean=%.6g worst=%.6g dt=%.2fs\n', gen, best_score, mean_score, worst_score, iter_time);
        end

        if ~exist('prev_best_score','var'), prev_best_score = best_score; end
        if best_score < prev_best_score - tol
            prev_best_score = best_score;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end
        if no_improve >= patience
            if verbose, fprintf('DE early stopping at gen %d (no improvement for %d gens)\n', gen, no_improve); end
            history = trim_history(history, gen);
            break;
        end
    end

    best.params = [round(best_ind(1)), best_ind(2)];
    best.rmse = best_score;
    best.evals = eval_count;
end

%% -------------------- GWO (modified) --------------------
function [best, history] = gwo_optimize_cmilk(data_test_missing_sets, ground_truth, num_iterations, opts)
    if nargin < 4, opts = struct(); end
    NP = getopt(opts,'PopSize',40);
    GMaxIters = getopt(opts,'MaxIters',60);
    MaxK = getopt(opts,'MaxK',50);
    EpsBounds = getopt(opts,'EpsBounds',[1e-8,1e-1]);
    use_parallel = getopt(opts,'UseParallel', false);
    verbose = getopt(opts,'Verbose', false);
    patience = getopt(opts,'EarlyStopPatience', 10);
    tol = getopt(opts,'EarlyStopTol', 1e-6);


    if use_parallel
        try p =gcp('nocreate');
            if isempty(p)
                use_parallel = false;
            end
        catch
            use_parallel = false;
        end
    end

    k_lb = 1; k_ub = MaxK;
    eps_lb = EpsBounds(1); eps_ub = EpsBounds(2);

    rng('shuffle');
    wolves = zeros(NP,2);
    wolves(:,1) = k_lb + (k_ub-k_lb).*rand(NP,1);
    wolves(:,2) = eps_lb + (eps_ub-eps_lb).*rand(NP,1);

    history.best_rmse_per_iter = nan(GMaxIters,1);
    history.mean_score_per_iter = nan(GMaxIters,1);
    history.worst_score_per_iter = nan(GMaxIters,1);
    history.time_per_iter = nan(GMaxIters,1);
    history.cum_time = nan(GMaxIters,1);
    history.evals_per_iter = nan(GMaxIters,1);
    history.time_per_eval_mean = nan(GMaxIters,1);
    history.time_per_eval_sum = nan(GMaxIters,1);
    history.cum_evals = nan(GMaxIters,1);

    eval_count = 0;
    cumtime = 0;
    no_improve = 0;
    best_score = inf;
    best_idx = 1;

    for gen = 1:GMaxIters
        if verbose, fprintf('GWO: gen %d/%d\n', gen, GMaxIters); end
        t0 = tic;

        scores = inf(NP,1);
        eval_times = nan(NP,1);
        if use_parallel
            parfor i = 1:NP
                params_eval = [round(max(1, wolves(i,1))), wolves(i,2)];
                params_eval(1) = min(max(params_eval(1), k_lb), k_ub);
                params_eval(2) = min(max(params_eval(2), eps_lb), eps_ub);
                [s, dt] = timed_eval(params_eval, data_test_missing_sets, ground_truth, num_iterations, false);
                scores(i) = s;
                eval_times(i) = dt;
            end
        else
            for i = 1:NP
                params_eval = [round(max(1, wolves(i,1))), wolves(i,2)];
                params_eval(1) = min(max(params_eval(1), k_lb), k_ub);
                params_eval(2) = min(max(params_eval(2), eps_lb), eps_ub);
                [s, dt] = timed_eval(params_eval, data_test_missing_sets, ground_truth, num_iterations, false);
                scores(i) = s;
                eval_times(i) = dt;
            end
        end
        
        eval_count = eval_count + NP;
        history.evals_per_iter(gen) = NP;
        history.time_per_eval_mean(gen) = mean(eval_times(isfinite(eval_times)));
        history.time_per_eval_sum(gen) = sum(eval_times(isfinite(eval_times)));
        history.cum_evals(gen) = eval_count;

        valid_idx = isfinite(scores);
        if any(valid_idx)
            mean_score = mean(scores(valid_idx));
            worst_score = max(scores(valid_idx));
        else
            mean_score = NaN; worst_score = NaN;
        end

        [scores_sorted, idx_sorted] = sort(scores);
        alpha = wolves(idx_sorted(1),:);
        beta  = wolves(idx_sorted(2),:);
        delta = wolves(idx_sorted(3),:);

        if scores_sorted(1) < best_score - tol
            best_score = scores_sorted(1);
            best_idx = idx_sorted(1);
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        a = 2 - 2*(gen-1)/(GMaxIters-1);

        new_wolves = wolves;
        for i = 1:NP
            X = wolves(i,:);
            r1 = rand(1,2); r2 = rand(1,2);
            A1 = 2*a.*r1 - a; C1 = 2.*r2; D_alpha = abs(C1.*alpha - X); X1 = alpha - A1.*D_alpha;
            r1 = rand(1,2); r2 = rand(1,2);
            A2 = 2*a.*r1 - a; C2 = 2.*r2; D_beta = abs(C2.*beta - X); X2 = beta - A2.*D_beta;
            r1 = rand(1,2); r2 = rand(1,2);
            A3 = 2*a.*r1 - a; C3 = 2.*r2; D_delta = abs(C3.*delta - X); X3 = delta - A3.*D_delta;
            Xnew = (X1 + X2 + X3)/3;
            Xnew(1) = min(max(Xnew(1), k_lb), k_ub);
            Xnew(2) = min(max(Xnew(2), eps_lb), eps_ub);
            new_wolves(i,:) = Xnew;
        end
        wolves = new_wolves;

        iter_time = toc(t0);
        cumtime = cumtime + iter_time;

        history.best_rmse_per_iter(gen) = best_score;
        history.mean_score_per_iter(gen) = mean_score;
        history.worst_score_per_iter(gen) = worst_score;
        history.time_per_iter(gen) = iter_time;
        history.cum_time(gen) = cumtime;

        if verbose
            fprintf('  gen %d -> best=%.6g mean=%.6g worst=%.6g dt=%.2fs\n', gen, best_score, mean_score, worst_score, iter_time);
        end

        if no_improve >= patience
            if verbose, fprintf('GWO early stopping at gen %d (no improvement for %d gens)\n', gen, no_improve); end
            history = trim_history(history, gen);
            break;
        end
    end

    best.params = [round(wolves(best_idx,1)), wolves(best_idx,2)];
    best.rmse = best_score;
    best.evals = eval_count;
end

function rmseScore = gaObjective_worker_safe(params, data_test_missing_sets, ground_truth, num_iterations, verbose)
    persistent localCache
    if isempty(localCache)
        localCache = struct(); 
    end

    tstart_all = tic;
    k_val = max(1, round(params(1))); 
    log_likelihood_threshold = params(2);

    key_raw = sprintf('k_%d_thr_%.12f', k_val, log_likelihood_threshold);
    key_field = matlab.lang.makeValidName(key_raw);

    if isfield(localCache, key_field)
        cached = localCache.(key_field);
        if verbose
            fprintf('[cache hit worker] %s -> RMSE=%.6f\n', key_raw, cached.agg_rmse);
        end
        rmseScore = cached.agg_rmse;
        return;
    end

    n_sets = size(data_test_missing_sets,1);
    set_rmses = nan(n_sets,1);
    elapsed_times = nan(n_sets,1);

    if verbose
        fprintf('GA eval starting: k=%d, threshold=%.8g on %d sets\n', k_val, log_likelihood_threshold, n_sets);
    end

    for set_idx = 1:n_sets
        t0 = tic;
        ds = data_test_missing_sets{set_idx, 1};   % dataset (table or numeric)
        if istable(ds)
            modified_dataset = table2array(ds);
        else
            modified_dataset = double(ds);
        end

        try
            imputed_dataset = optimize_k_and_threshold(k_val, log_likelihood_threshold, modified_dataset, num_iterations, verbose);
        catch ME
            warning('Imputer failed on set %d: %s. Returning NaN RMSE for this set.', set_idx, ME.message);
            imputed_dataset = modified_dataset;
        end

        missing_mask = isnan(modified_dataset);
        if ~any(missing_mask(:))
            set_rmse = 0;
        else
            gt_vals = ground_truth(missing_mask);
            pred_vals = imputed_dataset(missing_mask);
            if numel(gt_vals) ~= numel(pred_vals)
                warning('Size mismatch on set %d: ground-truth vs predicted counts differ. Setting NaN.', set_idx);
                set_rmse = NaN;
            else
                set_rmse = sqrt(mean((gt_vals - pred_vals).^2, 'omitnan'));
            end
        end

        elapsed_times(set_idx) = toc(t0);
        set_rmses(set_idx) = set_rmse;
        if verbose
            fprintf('  set %3d: RMSE=%.6f, time=%.3fs\n', set_idx, set_rmse, elapsed_times(set_idx));
        end
    end

    agg_rmse = mean(set_rmses(~isnan(set_rmses)));
    elapsed_total = toc(tstart_all);

    localCache.(key_field) = struct('agg_rmse', agg_rmse, 'set_rmses', set_rmses, 'elapsed_times', elapsed_times);

    if verbose
        fprintf('GA eval finished (worker): k=%d thr=%.8g -> aggregated RMSE=%.6f (total time=%.2fs)\n', k_val, log_likelihood_threshold, agg_rmse, elapsed_total);
    end

    rmseScore = agg_rmse;
end

function [imputed_dataset, logLikelihood] = optimize_k_and_threshold(k, log_likelihood_threshold, modified_dataset, max_iterations, verbose)

    [num_rows, num_cols] = size(modified_dataset);
    imputed_dataset = modified_dataset;
    imputed_datasets = cell(max_iterations,1);
    log_likelihoods = [];
    best_log_likelihood = -inf;

    num_iterations = min(max_iterations, 100);

    if verbose
        fprintf('    CMILK: start imputation (k=%d, thr=%.8g) rows=%d cols=%d\n', k, log_likelihood_threshold, num_rows, num_cols);
    end

    for iteration = 1:num_iterations
        if iteration == 1
            current_dataset = imputed_dataset;
        else
            current_dataset = imputed_datasets{iteration - 1};
        end

        updated_dataset = current_dataset;

        correlation_table = cell(num_cols, num_cols);
        coefficient_table = cell(num_cols, num_cols);
        for col1 = 1:num_cols
            for col2 = 1:num_cols
                if col1 == col2
                    correlation_table{col1,col2} = NaN;
                    coefficient_table{col1,col2} = NaN;
                    continue;
                end
                valid_rows = ~isnan(updated_dataset(:,col1)) & ~isnan(updated_dataset(:,col2));
                if sum(valid_rows) > 1
                    try
                        correlation_table{col1,col2} = corr(updated_dataset(valid_rows, col1), updated_dataset(valid_rows, col2));
                        coefficient_table{col1,col2} = mean(updated_dataset(valid_rows, col1) ./ updated_dataset(valid_rows, col2), 'omitnan');
                    catch
                        correlation_table{col1,col2} = NaN;
                        coefficient_table{col1,col2} = NaN;
                    end
                else
                    correlation_table{col1,col2} = NaN;
                    coefficient_table{col1,col2} = NaN;
                end
            end
        end

        if verbose
            fprintf('      iteration %d: computed corr/coeff tables\n', iteration);
        end

        if iteration == 1
            for row = 1:num_rows
                current_row = updated_dataset(row,:);
                modified_row = modified_dataset(row,:);
                missing_indices = find(isnan(modified_row));
                if isempty(missing_indices), continue; end
                available_indices = find(~isnan(modified_row));
                if isempty(available_indices), continue; end

                for missing_idx = missing_indices
                    correlations = cell2mat(correlation_table(missing_idx, :));
                    coefficients = cell2mat(coefficient_table(missing_idx, :));
                    available_correlations = correlations(available_indices);
                    available_coefficients = coefficients(available_indices);
                    if ~isempty(available_correlations)
                        [~, best_idx_local] = max(abs(available_correlations));
                        best_available_idx = available_indices(best_idx_local);
                        best_coefficient = available_coefficients(best_idx_local);
                        predicted_value = best_coefficient * current_row(best_available_idx);
                        updated_dataset(row, missing_idx) = predicted_value;
                    end
                end
            end
        else
            for row = 1:num_rows
                current_row = updated_dataset(row,:);
                modified_row = modified_dataset(row,:);
                missing_indices = find(isnan(modified_row));
                if isempty(missing_indices), continue; end
                available_indices = find(~isnan(modified_row));
                if isempty(available_indices), continue; end

                for missing_idx = missing_indices
                    correlations = cell2mat(correlation_table(missing_idx, :));
                    coefficients = cell2mat(coefficient_table(missing_idx, :));
                    available_correlations = correlations(available_indices);
                    available_coefficients = coefficients(available_indices);
                    if ~isempty(available_correlations)
                        [~, best_idx_local] = max(abs(available_correlations));
                        best_available_idx = available_indices(best_idx_local);
                        best_coefficient = available_coefficients(best_idx_local);
                        predicted_value = best_coefficient * current_row(best_available_idx);

                        target_column = modified_dataset(:, missing_idx);
                        observed_values = target_column(~isnan(target_column));

                        if ~isempty(observed_values)
                            dists = sqrt((observed_values - predicted_value).^2);
                            k_use = min(k, numel(dists));
                            [~, idxs] = mink(dists, k_use);
                            candidate_values = observed_values(idxs);
                            mu = mean(candidate_values);
                            sigma = std(candidate_values, 'omitnan');
                            if isnan(sigma) || sigma == 0
                                best_candidate_value = mean(candidate_values);
                            else
                                gauss_like = normpdf(candidate_values, mu, sigma);
                                [~, maxi] = max(gauss_like);
                                best_candidate_value = candidate_values(maxi);
                            end
                            updated_dataset(row, missing_idx) = best_candidate_value;
                        else
                            updated_dataset(row, missing_idx) = predicted_value;
                        end
                    end
                end
            end
        end

        imputed_values = updated_dataset(isnan(modified_dataset));
        if isempty(imputed_values)
            mu = 0; sigma = 1; log_likelihood = -Inf;
        else
            mu = mean(imputed_values, 'omitnan');
            sigma = std(imputed_values, 'omitnan');
            try
                log_likelihood = -normlike([mu, sigma], imputed_values);
            catch
                if sigma == 0 || isnan(sigma), sigma = 1e-6; end
                ll = sum(log(normpdf(imputed_values, mu, sigma) + eps));
                log_likelihood = ll;
            end
        end
        log_likelihoods = [log_likelihoods; log_likelihood];

        if verbose
            fprintf('      iteration %d: logL=%.6g\n', iteration, log_likelihood);
        end

        if iteration > 1
            log_likelihood_change = log_likelihoods(end) - log_likelihoods(end-1);
            if log_likelihoods(end) > best_log_likelihood
                best_log_likelihood = log_likelihoods(end);
            end
            if log_likelihood_change < log_likelihood_threshold
                if iteration-1 >= 1
                    updated_dataset = imputed_datasets{iteration-1};
                end
                imputed_datasets{iteration} = updated_dataset;
                if verbose
                    fprintf('      Converged at iteration %d (change=%.6g < thr=%.6g).\n', iteration, log_likelihood_change, log_likelihood_threshold);
                end
                break;
            end
        end

        imputed_datasets{iteration} = updated_dataset;
    end

    imputed_dataset = updated_dataset;
    logLikelihood = log_likelihoods(end);
end

% Correct signature: three outputs
%% ------------------ ga_logger OutputFcn (correct API) ------------------
function [state, options, optchanged] = ga_logger(options, state, flag, verbose)
    optchanged = false;
    global GA_BEST_PARAMS GA_BEST_RMSE GA_MEAN_RMSE GA_WORST_RMSE GA_POP_LOG GA_TIC GA_GEN_TIMES GA_GEN_DELTA GA_TOTAL_TIME

    try
        switch flag
            case 'init'
                if verbose, fprintf('ga_logger: init\n'); end
                % initialize if empty
                if isempty(GA_BEST_PARAMS), GA_BEST_PARAMS = []; end
                if isempty(GA_BEST_RMSE), GA_BEST_RMSE = []; end
                if isempty(GA_MEAN_RMSE), GA_MEAN_RMSE = []; end
                if isempty(GA_WORST_RMSE), GA_WORST_RMSE = []; end
                if isempty(GA_POP_LOG), GA_POP_LOG = struct('Generation', {}, 'Population', {}, 'Scores', {}); end
                if isempty(GA_GEN_TIMES), GA_GEN_TIMES = []; end
                if isempty(GA_GEN_DELTA), GA_GEN_DELTA = []; end
                % set timer if not set
                if isempty(GA_TIC)
                    GA_TIC = tic;
                end

            case 'iter'
                gen = [];
                if isfield(state,'Generation'), gen = state.Generation; end
                pop = [];
                scores = [];
                if isfield(state,'Population')
                    pop = state.Population;
                end
                if isfield(state,'Score')
                    scores = state.Score;
                elseif isfield(state,'Scores')
                    scores = state.Scores;
                end
                if ~isempty(scores) && ~isempty(pop)
                    valid_idx = isfinite(scores);
                    if any(valid_idx)
                        cur_best_rmse = min(scores(valid_idx));
                        cur_mean_rmse = mean(scores(valid_idx));
                        cur_worst_rmse = max(scores(valid_idx));
                        [~, best_idx] = min(scores);
                        cur_best_ind = pop(best_idx,:);
                    else
                        cur_best_rmse = NaN; cur_mean_rmse = NaN; cur_worst_rmse = NaN; cur_best_ind = [NaN,NaN];
                    end
                elseif isfield(state,'Best') && ~isempty(state.Best)
                    cur_best_ind = state.Best;
                    if isfield(state,'BestScore'), cur_best_rmse = state.BestScore; else cur_best_rmse = NaN; end
                    cur_mean_rmse = NaN; cur_worst_rmse = NaN;
                else
                    cur_best_rmse = NaN; cur_mean_rmse = NaN; cur_worst_rmse = NaN; cur_best_ind = [NaN,NaN];
                end
                if ~isempty(cur_best_ind) && numel(cur_best_ind) >= 2
                    k_val = round(max(1, cur_best_ind(1)));
                    eps_val = cur_best_ind(2);
                else
                    k_val = NaN; eps_val = NaN;
                end
                GA_BEST_PARAMS = [GA_BEST_PARAMS; [k_val, eps_val]];
                GA_BEST_RMSE   = [GA_BEST_RMSE; cur_best_rmse];
                GA_MEAN_RMSE   = [GA_MEAN_RMSE; cur_mean_rmse];
                GA_WORST_RMSE  = [GA_WORST_RMSE; cur_worst_rmse]; 
                gp.Generation = gen;
                gp.Population = pop;
                gp.Scores = scores;
                GA_POP_LOG(end+1) = gp;
                if ~isempty(GA_TIC)
                    cumt = toc(GA_TIC);
                else
                    cumt = NaN;
                end
                if isempty(GA_GEN_TIMES)
                    prev = 0;
                else
                    prev = GA_GEN_TIMES(end);
                end
                GA_GEN_TIMES = [GA_GEN_TIMES; cumt]; 
                GA_GEN_DELTA = [GA_GEN_DELTA; (cumt - prev)]; 

                if verbose
                    fprintf('  [ga_logger] gen=%s best=%.6g mean=%.6g worst=%.6g k~%g eps=%.6g cumtime=%.2fs dt=%.2fs\n', ...
                        mat2str(gen), cur_best_rmse, cur_mean_rmse, cur_worst_rmse, k_val, eps_val, cumt, (cumt - prev));
                end

            case 'done'
                if ~isempty(GA_TIC)
                    GA_TOTAL_TIME = toc(GA_TIC);
                else
                    GA_TOTAL_TIME = NaN;
                end
                if verbose, fprintf('ga_logger: done (total_time=%.2fs)\n', GA_TOTAL_TIME); end

            otherwise
        end
    catch 
        warning('ga_logger internal error: %s');
    end
end

function [score, elapsed] = timed_eval(params, data_test_missing_sets, ground_truth, num_iterations, verbose)
    if nargin < 5, verbose = false; end
    t0 = tic;
    try
        score = gaObjective_worker_safe(params, data_test_missing_sets, ground_truth, num_iterations, verbose);
    catch ME
        score = inf;
        if verbose
            warning('timed_eval: objective failed for params [%s]: %s', num2str(params), ME.message);
        end
    end
    elapsed = toc(t0);
end

function score = timed_eval_score(params, data_test_missing_sets, ground_truth, num_iterations, verbose)
    if nargin < 5, verbose = false; end
    [s, ~] = timed_eval(params, data_test_missing_sets, ground_truth, num_iterations, verbose);
    score = s;
end

function val = getopt(opts, name, default)
    if isfield(opts, name), val = opts.(name); else val = default; end
end

function history = trim_history(history, last_idx)
    fn = fieldnames(history);
    for f = 1:numel(fn)
        v = history.(fn{f});
        if isnumeric(v) || islogical(v)
            if isvector(v)
                history.(fn{f}) = v(1:min(last_idx,numel(v)));
            elseif size(v,1) >= last_idx
                history.(fn{f}) = v(1:last_idx, :);
            end
        elseif iscell(v)
            history.(fn{f}) = v(1:min(last_idx,numel(v)));
        end
    end
end
function v = safe_vec(s, name)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = [];
    end
end

function imputed_dataset = run_cmilk_single(modified_dataset, k_val, eps_val, num_cols, num_rows, threshold_ga, max_iterations)
    if nargin < 8
        max_iterations = 100;
    end
    imputed_dataset = modified_dataset;
    imputed_datasets = cell(max_iterations, 1);
    log_likelihoods = [];
    best_log_likelihood = -inf;

    for iteration = 1:max_iterations
        fprintf('Iteration %d...\n', iteration);
        if iteration == 1
            current_dataset = imputed_dataset;
        else
            current_dataset = imputed_datasets{iteration-1};
        end
        updated_dataset = current_dataset;
        correlation_table = cell(num_cols, num_cols);
        coefficient_table  = cell(num_cols, num_cols);

        for col1 = 1:num_cols
            for col2 = 1:num_cols
                if col1 ~= col2
                    valid_rows = ~isnan(updated_dataset(:, col1)) & ~isnan(updated_dataset(:, col2));
                    if sum(valid_rows) > 1
                        correlation_table{col1, col2} = corr(updated_dataset(valid_rows, col1), ...
                                                             updated_dataset(valid_rows, col2));
                        coefficient_table{col1, col2} = mean(updated_dataset(valid_rows, col1) ./ ...
                                                             (updated_dataset(valid_rows, col2) + eps_val), 'omitnan');
                    else
                        correlation_table{col1, col2} = NaN;
                        coefficient_table{col1, col2} = NaN;
                    end
                else
                    correlation_table{col1, col2} = NaN;
                    coefficient_table{col1, col2} = NaN;
                end
            end
        end

        for row = 1:num_rows
            current_row = updated_dataset(row, :);
            modified_row = modified_dataset(row, :);
            missing_indices = find(isnan(modified_row));
            for missing_idx = missing_indices
                correlations = cell2mat(correlation_table(missing_idx, :));
                coefficients  = cell2mat(coefficient_table(missing_idx, :));
                available_indices = find(~isnan(modified_row));
                if isempty(available_indices)
                    continue;
                end
                available_correlations = correlations(available_indices);
                available_coefficients = coefficients(available_indices);

                if any(~isnan(available_correlations))
                    [~, best_idx] = max(abs(available_correlations));
                    best_available_idx = available_indices(best_idx);
                    best_coefficient = available_coefficients(best_idx);
                    predicted_value = best_coefficient * current_row(best_available_idx);
                    target_column = modified_dataset(:, missing_idx);
                    observed_values = target_column(~isnan(target_column));
                    if ~isempty(observed_values)
                        distances = sqrt((observed_values - predicted_value).^2);
                        kk = min(k_val, numel(observed_values));
                        [~, closest_idx] = mink(distances, kk);
                        candidate_values = observed_values(closest_idx);

                        mu = mean(candidate_values);
                        sigma = std(candidate_values, 'omitnan');
                        sigma = max(sigma, eps_val);

                        gaussian_likelihoods = normpdf(candidate_values, mu, sigma);
                        [~, max_idx] = max(gaussian_likelihoods);
                        best_candidate_value = candidate_values(max_idx);
                        updated_dataset(row, missing_idx) = best_candidate_value;
                    else
                        updated_dataset(row, missing_idx) = predicted_value;
                    end
                else
                end
            end
        end 
        imputed_values = updated_dataset(isnan(modified_dataset));
        if ~isempty(imputed_values)
            mu = mean(imputed_values, 'omitnan');
            sigma = std(imputed_values, 'omitnan');
            if sigma <= 0 || isnan(sigma)
                log_likelihood = -inf;
            else
                log_likelihood = -normlike([mu, sigma], imputed_values);
            end
        else
            log_likelihood = -inf;
        end
        log_likelihoods = [log_likelihoods; log_likelihood];
        fprintf('Iteration %d: Log-likelihood = %.6f\n', iteration, log_likelihood);

        if iteration > 1
            if (log_likelihoods(end) - log_likelihoods(end-1)) < threshold_ga
                if ~isempty(imputed_datasets{iteration-1})
                    updated_dataset = imputed_datasets{iteration-1};
                end
                break;
            end
        end
        imputed_datasets{iteration} = updated_dataset;
    end 

    imputed_dataset = updated_dataset;
end