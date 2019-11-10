%Autor: Denilson Gomes Vaz da Silva
%Graduando em Engenharia da Computação
%Reconhecimento de Padroes

clear %limpa as variaveis
clc %limpa o visor 
close all %fecha as figuras

% Gerando o dataset
% Numero de amostras
N = 1000; %total
N1 = N; %classe 1
N2 = N; %classe 2
N3 = N; %classe 3

%--- Classe 1 -----
X1 = [1.0; 0.0; 0.5]*ones(1,N) + randn(3,N)*sqrt(0.02);

%--- Classe 2 -----
X2 = [0.5; 1.0; 0.0]*ones(1,N) + randn(3,N)*sqrt(0.02);

%--- Classe 3 -----
X3 = [0.0; 0.5; 1.0]*ones(1,N) + randn(3,N)*sqrt(0.02);

%Media de cada classe
u1 = mean(X1,2);
u2 = mean(X2,2);
u3 = mean(X3,2);

%Media geral
u = (u1 + u2 + u3)/3;

% Gráfico de dispersão
figure,  plot3(X1(1,:),X1(2,:),X1(3,:),'rd')
hold on
plot3(X2(1,:),X2(2,:),X2(3,:),'bd')
plot3(X3(1,:),X3(2,:),X3(3,:),'kd')
plot3(u1(1),u1(2),u1(3),'rx')
plot3(u2(1),u2(2),u2(3),'bx')
plot3(u3(1),u3(2),u3(3),'kx')
grid on

% Matriz de Covariança de cada classe
S1 = cov(X1.');
S2 = cov(X2.');
S3 = cov(X3.');

% Within-class scatter matrix
Sw = S1 + S2 + S3;

% Between-class scatter matrix
SB1 = N1*(u1 - u)*(u1 - u).';
SB2 = N2*(u2 - u)*(u2 - u).';
SB3 = N3*(u3 - u)*(u3 - u).';
SB = SB1 + SB2 + SB3;

% Eigendecomposition 
[V,D] = eig(inv(Sw)*SB)

% Vetores de projeção ótimos
w1 = V(:,3); %Autovetor com maior autovalor
w2 = V(:,2); %Autovetor com segundo maior autovalor
hold on, plot3([-w1(1) w1(1)]*2,[-w1(2) w1(2)]*2,[-w1(3) w1(3)]*2,'b-')
hold on, plot3([-w2(1) w2(1)]*2,[-w2(2) w2(2)]*2,[-w2(3) w2(3)]*2,'k-')

% Amostras projetadas (resultado do LDA)
y1(1,:) = w1.'*X1;
y2(1,:)  = w1.'*X2;
y3(1,:)  = w1.'*X3;
y1(2,:) = w2.'*X1;
y2(2,:)  = w2.'*X2;
y3(2,:)  = w2.'*X3;

% Histograma das amostras projetadas
%na primeira reta
figure,histogram(y1(1,:),30)
hold on
histogram(y2(1,:),30)
histogram(y3(1,:),30)

%na segunda reta
figure,histogram(y1(2,:),30)
hold on
histogram(y2(2,:),30)
histogram(y3(2,:),30)

% Matriz de correlação dos atributos
corrcoef([y1(1,:)' y1(2,:)' y2(1,:)' y2(2,:)' y3(1,:)' y3(2,:)'])

%---- k-Fold ----
k=10;

%--- Acertos ---
acertos1Total = 0;
acertos2Total = 0;
acertos3Total = 0;

for f=1:k
    %amostras de teste de cada classe
    teste1 = y1(:,1 + (f-1)*N/k : f*N/k);
    teste2 = y2(:,1 + (f-1)*N/k : f*N/k);
    teste3 = y3(:,1 + (f-1)*N/k : f*N/k);
    teste = [teste1 teste2 teste3]; % vetor de testes
    
    %amostras de treino
    treino1 = y1;
    treino1(:,1 + (f-1)*100 : f*100) = [];
    treino2 = y2;
    treino2(:,1 + (f-1)*100 : f*100) = [];
    treino3 = y3;
    treino3(:,1 + (f-1)*100 : f*100) = [];
        
    %Naive Bayes 
    %(múltiplas classes equiprováveis)
    %(atributos contínuos, gaussianos e descorrelacionados)
    %zerando os acertos
    acertos1 = 0;
    acertos2 = 0;
    acertos3 = 0;
       
    % Calculando vetor medio de cada classes
    sm1 = mean(y1,2);
    sm2 = mean(y2,2);
    sm3 = mean(y3,2);
    
    % matriz covariança de cada classe
    var1 = corrcoef([y1(1,:)' y1(2,:)']);
    var2 = corrcoef([y2(1,:)' y2(2,:)']);
    var3 = corrcoef([y3(1,:)' y3(2,:)']);
    
    for i=1:length(teste) %Para cada teste
        % Calculo para escolher a classe
        palpite1 = log(det(var1)) + (teste(:,i) - sm1)'*inv(var1)*(teste(:,i) - sm1);
        palpite2 = log(det(var2)) + (teste(:,i) - sm2)'*inv(var2)*(teste(:,i) - sm2);
        palpite3 = log(det(var3)) + (teste(:,i) - sm3)'*inv(var3)*(teste(:,i) - sm3);
        % As classes são equiprovaveis
        
        % Se o modelo escolher a classe 1 e o elemento seja da classe 1
        if palpite1 < palpite2 && palpite1 < palpite3 && i <= length(teste1)
            acertos1 = acertos1 + 1; %acerto da classe 1
        end
        % Se o modelo escolher a classe 2 e o elemento seja da classe 2
        if palpite2 < palpite1 && palpite2 < palpite3  && i > length(teste1) && i <= 2*length(teste1)
            acertos2 = acertos2 + 1; %acerto da classe 2
        end
        % Se o modelo escolher a classe 3 e o elemento seja da classe 3
        if palpite3 < palpite2 && palpite3 < palpite1  && i > 2*length(teste1) && i <= 3*length(teste1)
            acertos3 = acertos3 + 1; %acerto da classe 3
        end
    end
    acertos1Total = acertos1Total + acertos1; %acerto classe 1 total
    acertos2Total = acertos2Total + acertos2; %acerto classe 2 total
    acertos3Total = acertos3Total + acertos3; %acerto classe 3 total    
end
%printa os resultados das predições
disp('Acurácias obtidas com 10-Fold do Naive Bayes'); %exibe a mensagem acima
str = ['Taxa de acerto da classe1: ' num2str(acertos1Total/N)];
disp(str); %exibe a mensagem acima
str = ['Taxa de acerto da classe2: ' num2str(acertos2Total/N)];
disp(str); %exibe a mensagem acima
str = ['Taxa de acerto da classe3: ' num2str(acertos3Total/N)];
disp(str); %exibe a mensagem acima