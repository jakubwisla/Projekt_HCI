clear all;
close all;

% ścieżka do folderu zawierającego pliki audio
folder_path = '';

% pobranie listy plików audio z folderu
audio_files = dir(fullfile(folder_path, '*.wav'));

% pętla po wszystkich plikach audio
for i = 1:length(audio_files)
    % wczytanie pliku audio
    audio_path = fullfile(folder_path, audio_files(i).name);
    [input_audio, Fs] = audioread(audio_path);

    % losowa zmiana tonacji
    tonacja = 2^((-3 + rand()*6)/12);
    audio_pitch_shifted = interp1(0:1/Fs:(length(input_audio)-1)/Fs, input_audio, ...
        0:1/(tonacja*Fs):(length(input_audio)-1)/(tonacja*Fs));

    % losowa zmiana amplitudy
    amplituda = 0.5 + rand();
    audio_amplified = amplituda * audio_pitch_shifted;

    % losowe generowanie szumu
    szum = 0.02 * randn(size(audio_amplified));
    audio_noisy = audio_amplified + szum;

    % losowa zmiana prędkości odtwarzania
    przyspieszenie = 0.9 + rand()*0.2;
    nowa_czestotliwosc = Fs * przyspieszenie;
    
    % interpolacja dla zmiany prędkości odtwarzania
    audio_przyspieszone = resample(audio_noisy, round(nowa_czestotliwosc), Fs);
    
    % przycięcie do jednej sekundy, jeśli trwa dłużej
    if length(audio_przyspieszone) > Fs
        audio_przyspieszone = audio_przyspieszone(1:Fs);
    end
    
    % % odtworzenie zmodyfikowanego dźwięku
    % sound(audio_przyspieszone, Fs);

    % zapisanie zmodyfikowanego pliku audio
    new_file_path = fullfile(folder_path, ['aug_', audio_files(i).name]);
    audiowrite(new_file_path, audio_przyspieszone, Fs);
end