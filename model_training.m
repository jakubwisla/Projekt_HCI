clear all;
close all;

speedupExample = false; % skrócenie treningu
rng default; % zapewnienie powtarzalności wyników

% załadowanie danych
dataFolder = 'C:\Users\Krystian\Desktop\Interfejsy człowiek-komputer\L7\';
dataset = 'C:\Users\Krystian\Desktop\Interfejsy człowiek-komputer\L7\google_speech';

% tworzenie zbioru treningowego danych
ads = audioDatastore(fullfile(dataset,"train"), ...
    IncludeSubfolders=true, ...
    FileExtensions=".wav", ...
    LabelSource="foldernames");

commands = categorical(["lewo","prawo","gora","dol"]); % wykrywane komendy
background = categorical("background"); % tło

isCommand = ismember(ads.Labels,commands);
isBackground = ismember(ads.Labels,background);
isUnknown = ~(isCommand|isBackground); %ustalenie klasy 'unknown'

includeFraction = 0.1; % ilość danych 'unknown' użyta do treningu
idx = find(isUnknown);
idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
isUnknown(idx) = false;
ads.Labels(isUnknown) = categorical("unknown");

adsTrain = subset(ads,isCommand|isUnknown|isBackground); % ustalenie zbioru treningowego
adsTrain.Labels = removecats(adsTrain.Labels);

% tworzenie zbioru walidacyjnego danych
ads = audioDatastore(fullfile(dataset,"validation"), ...
    IncludeSubfolders=true, ...
    FileExtensions=".wav", ...
    LabelSource="foldernames");

isCommand = ismember(ads.Labels,commands);
isBackground = ismember(ads.Labels,background);
isUnknown = ~(isCommand|isBackground);

includeFraction = 0.3; % ilość danych 'unknown' użyta do walidacji
idx = find(isUnknown);
idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
isUnknown(idx) = false;
ads.Labels(isUnknown) = categorical("unknown");

adsValidation = subset(ads,isCommand|isUnknown|isBackground);
adsValidation.Labels = removecats(adsValidation.Labels);

% wyrysowanie rozkładu danych treningowych i walidacyjnych
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5])

tiledlayout(2,1)

nexttile
histogram(adsTrain.Labels)
title("Training Label Distribution")
ylabel("Number of Observations")
grid on

nexttile
histogram(adsValidation.Labels)
title("Validation Label Distribution")
ylabel("Number of Observations")
grid on

% wykonywane w przypadku skróconego treningu
if speedupExample
    numUniqueLabels = numel(unique(adsTrain.Labels)); %#ok<UNRCH> 
    % redukcja wielkości zbioru danych
    adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels / 20));
    adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files) / numUniqueLabels / 20));
end

% warunkowe użycie karty graficznej do obliczenia równoległego
if canUseParallelPool && ~speedupExample
    useParallel = true;
    gcp;
else
    useParallel = false;
end

% określenie parametrów analizy audio
fs = 16e3; % częstotliwość próbkowania zbioru danych - 16kHz

segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;

FFTLength = 512;
numBands = 50;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

% ekstrakcji cech z sygnału audio
afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    FFTLength=FFTLength, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);
setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);

% normalizacja sygnałów audio do tej samej długości
transform1 = transform(adsTrain,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
% wyodrebnienie cech audio
transform2 = transform(transform1,@(x)extract(afe,x));
% logarytmowanie danych
transform3 = transform(transform2,@(x){log10(x+1e-6)});

% przetworzenie danych treningowych i zapis w zmiennej XTrain
XTrain = readall(transform3,UseParallel=useParallel);
XTrain = cat(4,XTrain{:});
[numHops,numBands,numChannels,numFiles] = size(XTrain);

% ekstrakcja cech na danych walidacyjnych
transform1 = transform(adsValidation,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
XValidation = readall(transform3,UseParallel=useParallel);
XValidation = cat(4,XValidation{:});

% etykiety treningowe i walidacyjne
TTrain = adsTrain.Labels;
TValidation = adsValidation.Labels;

% definicja architektury modelu
classes = categories(TTrain);
classWeights = 1./countcats(TTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(classes);

timePoolSize = ceil(numHops/8);

dropoutProb = 0.5;
numF = 12;
layers = [
    imageInputLayer([numHops,afe.FeatureVectorLength])
    
    convolution2dLayer(3,numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,2*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([timePoolSize,1])
    dropoutLayer(dropoutProb)

    fullyConnectedLayer(numClasses)
    softmaxLayer];

% ustawienie opcji treningu
miniBatchSize = 32;
validationFrequency = floor(numel(TTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=5, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=validationFrequency, ...
    Metrics="accuracy");

% trening sieci neuronowej
trainedNet = trainnet(XTrain,TTrain,layers,@(Y,T)crossentropy(Y,T,classWeights(:),WeightsFormat="C"),options);

% ocena modelu sieci neuronowej
scores = minibatchpredict(trainedNet,XValidation);
YValidation = scores2label(scores,classes,"auto");
validationError = mean(YValidation ~= TValidation);
scores = minibatchpredict(trainedNet,XTrain);
YTrain = scores2label(scores,classes,"auto");
trainError = mean(YTrain ~= TTrain);

disp(["Training error: " + trainError*100 + " %";"Validation error: " + validationError*100 + " %"])

% macierz pomyłek dla danych walidacyjnych
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(TValidation,YValidation, ...
    Title="Confusion Matrix for Validation Data", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
sortClasses(cm,[commands,"unknown","background"])

% zapis modelu
filePath = 'nn_pl.mat'; % ścieżka do pliku
save(filePath, 'trainedNet', '-v7.3');