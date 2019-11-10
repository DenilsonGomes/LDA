%Autor: Denilson Gomes Vaz da Silva
%Graduando em Engenharia da Computação
%Reconhecimento de Padroes

clear %limpa as variaveis
clc %limpa o visor 
close all

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
w1 = V(:,2);
w2 = V(:,3);
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