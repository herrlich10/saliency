clear all; clc;

saliency1 = double(imread('road/saliency1.png'))/255;
saliency2 = double(imread('road/saliency2.png'))/255;
large = double(imread('road/large.png'))/255;
fixation = imread('road/fixation.png')>128;
other = imread('road/other.png')>128;

fprintf('AUC_Judd = %.8f\n', AUC_Judd(saliency1, fixation));
fprintf('AUC_Borji = %.8f\n', AUC_Borji(saliency1, fixation, 10000));
fprintf('AUC_shuffled = %.8f\n', AUC_shuffled(saliency1, fixation, other, 10000));
fprintf('NSS = %.8f\n', NSS(saliency1, fixation));
fprintf('CC = %.8f\n', CC(saliency1, saliency2));
fprintf('SIM = %.8f\n', similarity(saliency1, saliency2));
fprintf('EMD = %.8f\n', EMD(saliency1, saliency2));

fprintf('----------\n');
fprintf('AUC_Judd = %.8f\n', AUC_Judd(large, fixation));
fprintf('AUC_Borji = %.8f\n', AUC_Borji(large, fixation, 10000));
fprintf('AUC_shuffled = %.8f\n', AUC_shuffled(large, fixation, other, 10000));
fprintf('NSS = %.8f\n', NSS(large, fixation));
fprintf('CC = %.8f\n', CC(large, saliency2));
fprintf('SIM = %.8f\n', similarity(large, saliency2));
fprintf('EMD = %.8f\n', EMD(large, saliency2));
