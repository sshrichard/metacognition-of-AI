clear all
close all




%-------------------------------------------------------------------------%
% Interacting with ChatGPT using OpenAI API %
%-------------------------------------------------------------------------%
function text = chatgpt(userMessage, model)


    %--% ARGUMENTS %--%
    % userMessage = prompt (string)
    % model = Ex: "gpt-5" (string)



    %--% Safety tests %--%
    %---------------------------------------------------------------------%
    if or(nargin < 2, isempty(model)), model = 'gpt-4o'; end
    apiKey = getenv('OPENAI_API_KEY'); % API key defined below
    if isempty(apiKey)
        error("Set your API key");
    end
    %---------------------------------------------------------------------%



    %--% WHERE TO GO %--%
    url = 'https://api.openai.com/v1/chat/completions';
    


    %--% What do we send? %--%
    %---------------------------------------------------------------------%
    msgs = struct( ...
        'role',   {'system','user'}, ...
        'content',{'You are a helpful assistant.', char(userMessage)});
    payload = struct( ...
        'model', char(model), ...
        'messages', msgs); 
    %---------------------------------------------------------------------%



    %--% Let's interact with the web %--%
    %---------------------------------------------------------------------%
  
    import matlab.net.*
    import matlab.net.http.*
    headers = HeaderField('Authorization', ['Bearer ' char(apiKey)]);

    body = MessageBody(payload);
    req  = RequestMessage('post', headers, body);
    resp = req.send(URI(url));
    
     % If there is a problem when submitting the HTTP request
    if resp.StatusCode ~= matlab.net.http.StatusCode.OK
        errText = "<no details>";
        raw = resp.Body.Data;
        try % ... to extract the error content in a readable format
            if isa(raw,'uint8'), errText = char(raw);
            elseif ischar(raw),  errText = raw;
            elseif isstring(raw), errText = char(raw);
            elseif isstruct(raw)
                if isfield(raw,'error') && isfield(raw.error,'message')
                    errText = raw.error.message;
                else
                    errText = jsonencode(raw);
                end
            else
                errText = jsonencode(raw);
            end
        catch ME
            errText = sprintf('<failed to parse error body: %s>', ME.message);
        end
        error("OpenAI API error %s\nDetails: %s", string(resp.StatusLine), errText);
        % errText is the details while resp.StatusLine tells you at a glance the problem
    end
    %---------------------------------------------------------------------%
    %---------------------------------------------------------------------%
    data = resp.Body.Data;                  % already JSON-decoded struct
    text = data.choices(1).message.content; % first assistant message
    %---------------------------------------------------------------------%


end
%-------------------------------------------------------------------------%












%-------------------------------------------------------------------------%
% Loading dataset %
%-------------------------------------------------------------------------%

%--% Accessing the dataset %--%
dataset_name = 'SST-2';
numSamples = 1000; % number of sentences submitted to the charbot

file_name = "train-00000-of-00001.parquet";
testData = parquetread(file_name);

sentences = testData.sentence; % column "sentence"
labels    = testData.label;    % column "label"

idx = randperm(height(testData), numSamples); % we randomly select some entries
sampledSentences = sentences(idx);
sampledLabels    = labels(idx);
%------------------------------------------------------------------------%







%-------------------------------------------------------------------------%
% Prompt design %
%-------------------------------------------------------------------------%
base_PROMPT = fileread("prompt.txt");
%-------------------------------------------------------------------------%





%-------------------------------------------------------------------------%
% Asking chatGPT to perform the type-1 and type-2 taska %
%-------------------------------------------------------------------------%

 % API key defined as an environment variable
setenv('OPENAI_API_KEY','KEY');

for i = 1:numSamples
    
    sentence = sampledSentences{i};

    %--% Build the prompt with the final sentence appended %--%
    full_PROMPT = append(base_PROMPT, newline, newline, '"', sentence, '"');


    response = chatgpt(full_PROMPT, "gpt-5");

    % fprintf('--- Response %d ---\n%s\n', i, response); % (%d: decimal placeholder; %s: string placeholder)
    responses{i} = response ; % store responses in a cell array
end
%-------------------------------------------------------------------------%











%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
% Saving the data %
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%


%--% Creating the data folder %--%
outdir = "RESULTS";
if ~isfolder(outdir), mkdir(outdir); end
%-------------------------------------------------------------------------%


%-------------------------------------------------------------------------%
%--% Saving the experiment material and metadata %--%
%-------------------------------------------------------------------------%
sentences = string(sampledSentences); % convert to string array if not the case
labels    = string(sampledLabels); % convert to string array if not the case

% Open file and fill it
txt_filename = fullfile(outdir, "metadata.txt"); 
fid = fopen(txt_filename, 'a');
fprintf(fid, '# dataset: %s\n', dataset_name);
fprintf(fid, "model: gpt-5 \n");
fprintf(fid, '# file: %s\n', file_name);
fprintf(fid, '# n: %d\n', numSamples);
fprintf(fid, '# fields: index, sentence, label\n\n');
fprintf(fid, 'index\tsentence\tlabel\n'); % column header
for i = 1:numSamples
    fprintf(fid, '%d\t%s\t%s\n', idx(i), sentences(i), labels(i));
end
fclose(fid);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Saving chatGPT answers %
%-------------------------------------------------------------------------%
valence_pred    = zeros(numSamples,1);
confidence_pred = zeros(numSamples,1);

for i = 1:numSamples
    item = responses{i};

    % If the response is JSON text, decode it; if it's already a struct, use it
    if or(ischar(item), isstring(item))
        r = jsondecode(strtrim(string(item)));
    elseif isstruct(item)
        r = item;
    else
        error('Unexpected type in responses{%d}: %s', i, class(item));
    end

    % reminder: r = {"decision":"<0|1>","confidence":<1-10>} (imposed by the prompt)
    valence_pred(i)    = str2double(r.decision);   % "0"/"1" -> 0/1 (turns a string into a number)
    confidence_pred(i) = double(r.confidence);     % ensure numeric
end

% Create 2-column table with header
T = table(valence_pred, confidence_pred, ...
          'VariableNames', {'valence','confidence'});

% Save to CSV
csv_filename = fullfile(outdir, "chatgpt_responses.csv");
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
% Save the analysis in text file %
%-------------------------------------------------------------------------%
vec2str = @(v) strjoin(string(v), ' '); % make a list of strings out of a vector
fit_str   = evalc('disp(fit)');   % string representation of fit

txt_path = fullfile(outdir, "HMeta_d_results.txt");
fid = fopen(txt_path, 'a');

fprintf(fid, "Date: %s\n", datestr(datetime('now')));
fprintf(fid, "numSamples: %d\n\n", numSamples);
fprintf(fid, "model: gpt-5 \n");


fprintf(fid, "nR_S1      = [%s]\n", vec2str(nR_S1));
fprintf(fid, "nR_S2     = [%s]\n", vec2str(nR_S2));

fprintf(fid, "FIT (actual):\n%s\n", fit_str);

fclose(fid);

% essential outputs saved in a separated file
d1      = fit.d1;
meta_d  = fit.meta_d;
M_ratio = fit.M_ratio;

vals_path = fullfile(outdir, "meta_d_values.txt");
fid_vals = fopen(vals_path, 'a');
fprintf(fid_vals, "%f %f %f\n", d1, meta_d, M_ratio);
fclose(fid_vals);
%-------------------------------------------------------------------------%


