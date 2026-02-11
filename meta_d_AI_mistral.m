clear all
close all



%-------------------------------------------------------------------------%
% Loading dataset %
%-------------------------------------------------------------------------%
dataset_name = 'SST-2';
numSamples = 100; % number of sentences submitted to the charbot

file_name = "train-00000-of-00001.parquet";
testData = parquetread(file_name);

sentences = testData.sentence; % column "sentence"
labels    = testData.label;    % column "label"

idx = randperm(height(testData), numSamples); % we randomly select some entries
sampledSentences = sentences(idx);
sampledLabels    = labels(idx);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Prompt %
%-------------------------------------------------------------------------%
base_PROMPT = fileread("prompt_metad_A.txt");
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Asking the model to perform the type 1 and type 2 tasks %
%-------------------------------------------------------------------------%

setenv('MISTRAL_API_KEY','KEY'); % replace KEY by your API key 

% calling Mistral API with the argument "mistral-large-latest" invoked the model Large-Instruct-2411 at the time of our simulations
model_name = "mistral-large-latest";

responses = cell(numSamples,1);
for i = 1:numSamples

    %--% Build the prompt with the final sentence appended %--%
    sentence = sampledSentences{i};
    full_PROMPT = append(base_PROMPT, newline, newline, '"', sentence, '"');

    %--% Call the model %--%
    response = mistral(full_PROMPT, model_name);

    %--% Display and store the response %--%
    fprintf('--- Response %d ---\n%s\n', i, response);
    responses{i} = response; % store responses in a cell array
end
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
% Saving the data %
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%


%--% Creating the data folder %--%
outdir = "RESULTS_MISTRAL_task_A_sentiment_analysis";
if ~isfolder(outdir), mkdir(outdir); end
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
%--% Saving the submitted sentences and metadata %--%
%-------------------------------------------------------------------------%
sentences_str = string(sampledSentences);
labels_str    = string(sampledLabels);    

txt_filename = fullfile(outdir, "submitted_sentences_and_metadata.txt");
fid = fopen(txt_filename, 'a');
fprintf(fid, '# dataset: %s\n', dataset_name);
fprintf(fid, "model: %s \n", model_name);
fprintf(fid, '# file: %s\n', file_name);
fprintf(fid, '# n: %d\n', numSamples);
fprintf(fid, '# fields: index, sentence, label (ground truth)\n\n');
fprintf(fid, 'index\tsentence\tlabel\n'); % column header
for i = 1:numSamples
    fprintf(fid, '%d\t%s\t%s\n', idx(i), sentences_str(i), labels_str(i));
end
fclose(fid);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Saving the model's answers %
%-------------------------------------------------------------------------%
valence_pred    = zeros(numSamples,1);
confidence_pred = zeros(numSamples,1);

for i = 1:numSamples

    r = []; % ensures we don't save multiple times the same answer
    item = responses{i};

    if or(ischar(item), isstring(item))
        try
            r = jsondecode(regexprep(strtrim(string(item)), '```json|```', ''));
        catch ME
            error('Invalid JSON in responses{%d}: %s', i, ME.message);
        end
    elseif isstruct(item)
        r = item;
    else
        error('Unexpected type in responses{%d}: %s', i, class(item));
    end

    % reminder: r = {"decision":"<0|1>","confidence":<1-5>} (imposed by the prompt)
    valence_pred(i)    = str2double(r.decision);
    confidence_pred(i) = double(r.confidence);
end

% Create 2-column table with header
T = table(valence_pred, confidence_pred, ...
          'VariableNames', {'valence','confidence'});

% Save to CSV
csv_filename = fullfile(outdir, "mistral_responses.csv");
writetable(T, csv_filename, 'WriteMode', 'append');
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Computing meta-d' %
%-------------------------------------------------------------------------%
K  = 5; % confidence scale size

y_true = str2double(string(sampledLabels(:)));  
y_hat  = double(valence_pred(:));   
conf   = double(confidence_pred(:)); % round() should not do any work



%--% preparing data to feed Fleming's (see Fleming, 2017) %--%
cnt = @(mask, c) accumarray(c(mask), 1, [K, 1]);
nR_S1 = [flip(cnt(y_true==0 & y_hat==0, conf)); cnt(y_true==0 & y_hat==1, conf)].'; % [K,1] is implicit
nR_S2 = [flip(cnt(y_true==1 & y_hat==0, conf)); cnt(y_true==1 & y_hat==1, conf)].';

%--% Let's fit Fleming's Hierarchical Bayesian model %--%
fit  = fit_meta_d_mcmc(nR_S1,      nR_S2);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Save the SDT analysis in text file %
%-------------------------------------------------------------------------%
vec2str = @(v) strjoin(string(v), ' '); % make a string out of a vector's elements
fit_str   = evalc('disp(fit)');   % save everything that fit printed
txt_path = fullfile(outdir, "HMeta_d_results.txt");
fid = fopen(txt_path, 'a');
fprintf(fid, "Date: %s\n", datestr(datetime('now')));
fprintf(fid, "numSamples: %d\n\n", numSamples);
fprintf(fid, "model: %s \n", model_name);
fprintf(fid, "nR_S1     = [%s]\n", vec2str(nR_S1));
fprintf(fid, "nR_S2     = [%s]\n", vec2str(nR_S2));
fprintf(fid, "FIT (actual):\n%s\n", fit_str);
fclose(fid);

% essential outputs saved in a separated file
d1      = fit.d1;
meta_d  = fit.meta_d;
M_ratio = fit.M_ratio;
% no header to handle the case where multiple running script aims to 
% write within this file in order to produce a clean multi-row file
vals_path = fullfile(outdir, "meta_d_values.txt");
fid_vals = fopen(vals_path, 'a');
fprintf(fid_vals, "%f %f %f\n", d1, meta_d, M_ratio); 
fclose(fid_vals);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Interacting with Mistral using Mistral AI API %
%-------------------------------------------------------------------------%
function text = mistral(userMessage, model)

    %--% ARGUMENTS %--%
    % userMessage = prompt (string)
    % model
    

    %--% SAFETY CHECKS %--%
    %---------------------------------------------------------------------%
    % safety check for arguments
    if or(nargin < 2, isempty(model)), model = 'mistral-large-latest'; end

    % safety check for the API key
    apiKey = getenv('MISTRAL_API_KEY');
    if isempty(apiKey)
        error("Set API key");
    end
    %---------------------------------------------------------------------%


    %--% WHERE TO GO %--%
    url = 'https://api.mistral.ai/v1/chat/completions';


    %--% What do we send? %--%
    %---------------------------------------------------------------------%
    msg = struct( ...
        'role',   'user', ...
        'content', char(userMessage) ...
    );
    payload = struct( ...
        'model', char(model), ...
        'messages', {{msg}}, ...
        'temperature', 0, ...     
        'max_tokens', 4096 ...        
    );
    %---------------------------------------------------------------------%


    %--% Let's interact with the web %--%
    %---------------------------------------------------------------------%
    import matlab.net.*
    import matlab.net.http.*

    headers = [
        HeaderField('Authorization', ['Bearer ' char(apiKey)]), ...
        HeaderField('content-type', 'application/json')
    ];

    body = MessageBody(payload);
    req  = RequestMessage('post', headers, body);
    resp = req.send(URI(url));

     % If there is a problem when submitting the HTTP request, let's try to print the error
    if resp.StatusCode ~= matlab.net.http.StatusCode.OK
        raw = resp.Body.Data;
        if isstruct(raw) && isfield(raw,'error') && isfield(raw.error,'message')
            details = raw.error.message;
        else
            details = "no details";
        end
        error("API request failed (%s): %s", string(resp.StatusLine), details);
    end
    %---------------------------------------------------------------------%


    %--% Extract the model's response %--%
    %---------------------------------------------------------------------%
    data = resp.Body.Data;                 
    text = data.choices(1).message.content; % answer
    %---------------------------------------------------------------------%

end
%-------------------------------------------------------------------------%
