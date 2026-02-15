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
base_PROMPT = fileread("prompt_taskA_risk_configuration_0.txt");
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Asking the model to perform the task %
%-------------------------------------------------------------------------%

setenv('OPENAI_API_KEY','KEY'); % replace KEY by your API key 

model_name = "gpt-5";

responses = cell(numSamples, 1);

for i = 1:numSamples 
    
    %--% Build the prompt with the final sentence appended %--%
    sentence = sampledSentences{i};
    full_PROMPT = append(base_PROMPT, newline, newline, '"', sentence, '"');

    %--% Call the model %--%
    response = chatgpt(full_PROMPT, model_name);

     %--% Display and store the response %--%
    fprintf('--- Response %d ---\n%s\n', i, response); 
    responses{i} = response ; % store responses in a cell array
end
%-------------------------------------------------------------------------%











%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
% Saving the data %
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%


%--% Creating the data folder %--%
outdir = "RESULTS_GPT_c_calibration_task_A";
if ~isfolder(outdir), mkdir(outdir); end
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
%--% Saving the submitted sentences and metadata %--%
%-------------------------------------------------------------------------%
sentences_str = string(sampledSentences);
labels_str    = string(sampledLabels);    

txt_filename = fullfile(outdir, "submitted_sentences_and_metadata.txt");
fid = fopen(txt_filename, 'a');
fprintf(fid, "Date: %s\n", datestr(datetime('now')));
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

    % reminder: r = {"decision":"<0|1>"} (imposed by the prompt)
    valence_pred(i) = str2double(r.decision);   % "0"/"1" -> 0/1 (turns a string into a number)
end

% Create 1-column table with header
T = table(valence_pred, 'VariableNames', {'valence'});

% Save to CSV
csv_filename = fullfile(outdir, "chatgpt_responses.csv");
writetable(T, csv_filename, 'WriteMode', 'append');
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Computing c from FAR and HR %
%-------------------------------------------------------------------------%
y_true = str2double(string(sampledLabels(:))); 
y_hat = double(valence_pred(:));

% Counts
hits   = sum((y_true == 1) & (y_hat == 1));  % S2 trials called 1
misses = sum((y_true == 1) & (y_hat == 0));  % S2 trials called 0
FA     = sum((y_true == 0) & (y_hat == 1));  % S1 trials called 1
CR     = sum((y_true == 0) & (y_hat == 0));  % S1 trials called 0

% Rates
HR  = hits / max(1, hits + misses);          % hit rate
FAR = FA   / max(1, FA   + CR);              % false-alarm rate


% Compute z()
zHR  = norminv(HR);
zFAR = norminv(FAR);

% Compute d' and decision criterion c
dprime = zHR - zFAR;
c      = -0.5 * (zHR + zFAR);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Save this %
%-------------------------------------------------------------------------%
txt_filename = fullfile(outdir, "sdt_summary.txt");
fid = fopen(txt_filename, 'a');

% Write header
% fprintf(fid, 'hits\tmisses\tFA\tCR\tc\tdprime\n');

% Write data row
fprintf(fid, '%d\t%d\t%d\t%d\t%.6f\t%.6f\n', hits, misses, FA, CR, c, dprime);

fclose(fid);
%-------------------------------------------------------------------------%



%-------------------------------------------------------------------------%
% Interacting with ChatGPT using OpenAI API %
%-------------------------------------------------------------------------%
function text = chatgpt(userMessage, model)


    %--% ARGUMENTS %--%
    % userMessage = prompt (string)
    % model


    %--% Safety tests %--%
    %---------------------------------------------------------------------%
    % safety check for arguments
    if or(nargin < 2, isempty(model)), model = 'gpt-5'; end

    % safety check for the API key
    apiKey = getenv('OPENAI_API_KEY');
    if isempty(apiKey)
        error("Set API key");
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
