clear all;
close all;

% ustawienia parametrów nagrywania
Fs = 16000; % częstotliwość próbkowania
Duration = 1; % długość nagrania
Threshold = 0.05; % progowa wartość dla detekcji

% inicjalizacja obiektu nagrywania
recObj = audiorecorder(Fs, 16, 1);

disp('Rozpoczęcie nagrywania...');

i = 0;
while true
    disp('start');
    % nagrywanie dźwięku przez zdefiniowany czas
    recordblocking(recObj, Duration);
    
    % pobranie nagranego dźwięku
    audioData = getaudiodata(recObj);
    
    % sprawdzenie, czy wykryto mówienie
    energy = sum(audioData.^2); % obliczenie energii dźwięku
    if energy > Threshold
        disp('    wykryto');
        
        % zapis dźwięku do pliku .wav
        filename = ['dol3_', num2str(i) , '.wav'];
        i = i + 1;
        audiowrite(filename, audioData, Fs);
        
    else
        disp('Nie wykryto');
    end
end