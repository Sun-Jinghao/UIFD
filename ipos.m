
alpha =[
    0.018 0.046 0.334;
    0.022 0.047 0.334;
    0.024 0.047 0.334;
    0.024 0.047 0.334;
    0.039 0.051 0.336;
    0.105 0.068 0.344;
    0.154 0.078 0.346;
    0.297 0.127 1.78;
    0.542 0.233 0.403;
    0.943 0.43 0.456;
    ];
beta = [
    0.0038 0.0021 0.0009;
    0.0063 0.0040 0.0023;
    0.062 0.078 0.393;
    0.504 0.387 0.27;
    1.38 1.06 0.74;
    0.514 0.395 0.274;
    1.5 1.15 0.8;
    1.87 1.44 2.87;
    3.3 2.54 1.77;
    4.39 3.38 2.35;
    ];

for count=10:1:10
    D=10;
    B=0.55;
    a=count;
    C1= (exp(-alpha(a,1)*D));
    C2= (exp(-alpha(a,2)*D));
    C3= (exp(-alpha(a,3)*D));
    inputfile_1='nyu-depth-datasets/3/';
    inputfile_2='nyu-depth-datasets/1/';
    Files_1=dir([inputfile_1 '*.png']);
    Files_2=dir([inputfile_2 '*.png']);
    number=length(Files_1);
    disp(number);
    T =0.0*ones(480,640,3);
    for i=0:number-1
        img=imread([inputfile_2 num2str(i) '.png']);
        img_rgb=im2double(img);   
        depth_data=imread([inputfile_1 num2str(i) '.png']);
        %depth_data=imread([inputfile_1 Files_1(i).name]);
        %depth_img=mapminmax(depth_data,0,1);
        depth_img=im2double(depth_data);

        for x=1:480 
            for y=1:640 
                T(x, y, 1) =exp(-beta(a,1)* depth_img(x,y));
                T(x, y, 2) =exp(-beta(a,2)* depth_img(x,y));
                T(x, y, 3) =exp(-beta(a,3)* depth_img(x,y)) ;            
                img_rgb(x,y,3)=img_rgb(x,y,3)*C1*T(x,y,1)+(1-T(x,y,1))* 0.35;
                img_rgb(x,y,2)=img_rgb(x,y,2)*C2*T(x,y,2)+(1-T(x,y,2))* 0.95;
                img_rgb(x,y,1)=img_rgb(x,y,1)*C3*T(x,y,3)+(1-T(x,y,3))* 0.95;
            end
        end
        switch count
%             case 1
%                 imwrite(img_rgb,['Nyu/after/10m/1/',num2str(i),'.png']);
%      
%             case 2
%                 imwrite(img_rgb,['Nyu/after/10m/2/',num2str(i),'.png']);
%              
%             case 3
%                 imwrite(img_rgb,['Nyu/after/10m/3/',num2str(i),'.png']);
%             case 4
%                 imwrite(img_rgb,['Nyu/after/10m/4/',num2str(i),'.png']);
%             case 5    
%                 imwrite(img_rgb,['Nyu/after/10m/5/',num2str(i),'.png']);
%          
%             case 6    
%                 imwrite(img_rgb,['Nyu/after/10m/6/',num2str(i),'.png']);
%             case 7   
%                 imwrite(img_rgb,['Nyu/after/10m/7/',num2str(i),'.png']);
%                
%              case 8  
%                 imwrite(img_rgb,['Nyu/after/10m/8/',num2str(i),'.png']);
%           
%             case 9  
%                 imwrite(img_rgb,['Nyu/after/10m/9/',num2str(i),'.png']);
             
            case 10  
                imwrite(img_rgb,['Nyu/after/10m/10/',num2str(i),'.png']);
             
        end        
    end
    disp(count);
end
disp("over!!!")



