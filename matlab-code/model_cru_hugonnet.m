function model_cru_hugonnet(experiment_number)

%clear all
%experiment_number='1';

load_parameters;

parameters=parameters(str2num(experiment_number),:);

region_number=sprintf('%2.2i',parameters(1));

lapse_precip = parameters(2);     %  default in Marzeion et al., 2012: -3.0e-04;
temp_prec_solid = parameters(3);  %  default in Marzeion et al., 2012: 3;
temp_melt = parameters(4);        %  default in Marzeion et al., 2012: 1;
precip_factor =parameters(5);   %  default in Marzeion et al., 2012: 2.5;

%,lapse_precip,temp_prec_solid,temp_melt,precip_factor)

%clear all
%region_number='11';

%L_V_scaling_error=1;      % in percent/100 (i.e., 1 = 100 %)
%A_V_scaling_error=0.4;      % in percent/100 (i.e., 0.4 = 40 %)

%area_error_rgi=5;  % in percent

%tau_L_error=5;            % in percent/100 (i.e., 5 = 500 %)
%tau_A_error=5;            % in percent/100 (i.e., 5 = 500 %)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% read in rgi data

eval(['load rgi_v6/region' region_number '.mat;']);

nan_idx=zeros(size(rgi_area));

nan_idx(isnan(rgi_lat) | isnan(rgi_lon) | isnan(rgi_year) | isnan (rgi_zmax) | isnan(rgi_zmean) | isnan(rgi_zmin))=1;

rgi_year(rgi_year>2020)=2020;
rgi_year(rgi_year<1903)=1903;

%%% end read in rgi data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% start model

%load cross_validation_cru_new t_star_cross model_bias number_closest_glaciers length_of_climatology...
%    lon_mb lat_mb mse_mu_star_cross

%%% parameter values

length_of_climatology=31;

%lapse_precip = -4.0e-04;     %  default in Marzeion et al., 2012: -3.0e-04;
%temp_prec_solid = 4;  %  default in Marzeion et al., 2012: 3;
%temp_melt = 0;        %  default in Marzeion et al., 2012: 1;
%precip_factor =3;   %  default in Marzeion et al., 2012: 2.5;

cru_years=1901:2021;

gamma_glacier=1.375;                 % Bahr et al. 1997, The physical basis...
c_a_glacier=0.0340;                  % =0.191 m^(3-2gamma), Bahr 1997, Global distributions of glacier properties: A stochstic scaling paradigm
q_glacier=2.2;                       % Bahr et al. 1997, The physical basis...
c_l_glacier=0.018;                   % =4.55 m^(3-q), Radic, Hock, Oerlemans 2008: Analysis of scaling methods 

gamma_icecap=1.25;                   % Radic & Hock (2010), Regional and global volumes of glaciers derived from...
c_a_icecap=0.0538;                   % =1.7026 m^(3-2gamma), Radic & Hock (2010), Regional and global volumes of glaciers derived from...
q_icecap=2.5;                        % from H = 3.4/(1000^0.5) L^0.5, 3.4 (Patterson 1994) for L and H in m changed to km...
c_l_icecap=0.2252;                   % and V = 2/3 pi H L^2 

%mean_rmse=mean(mse_mu_star_cross.^(1/2));

time_cru=(1901+(1/24)):(1/12):(2022-(1/24)); 

height_cru_glacier=nan(1,length(rgi_area));
height_cru_clim_glacier=nan(1,length(rgi_area));
lapse_temp=nan(1,length(rgi_area));
rho_lapse_temp=nan(1,length(rgi_area));
p_lapse_temp=nan(1,length(rgi_area));
%dist=nan(length(rgi_area),length(lon_mb));
%mean_dist=nan(1,length(rgi_area));
%t_star_rgi=nan(1,length(rgi_area));
%model_bias_rgi=nan(1,length(rgi_area));
mu_rgi=nan(1,length(rgi_area));
temp_range=nan(1,length(rgi_area));
turnover=nan(1,length(rgi_area));
temp_glacier_tongue_clim=nan(1,12);
precip_glacier_clim=nan(1,12);
A=nan(length(rgi_area),length(time_cru)/12);
L=nan(length(rgi_area),length(time_cru)/12);
V=nan(length(rgi_area),length(time_cru)/12);
dL=nan(length(rgi_area),length(time_cru)/12);
dV=nan(length(rgi_area),length(time_cru)/12-1);
mb_modeled=nan(length(rgi_area),length(time_cru)/12-1);
%sust_water_delay=nan(length(rgi_area),length(time_cru)/12-1);
%mmp_sust=nan(length(rgi_area),length(time_cru)/12-1,12);
%mmp_total=nan(length(rgi_area),length(time_cru)/12-1,12);
mb_monthly_sust=nan(length(rgi_area),length(time_cru)/12-1,12);
mb_monthly_total=nan(length(rgi_area),length(time_cru)/12-1,12);
n_months_above_freezing=nan(length(rgi_area),length(time_cru)/12);
%A_error=nan(length(rgi_area),length(time_cru)/12);
%L_error=nan(length(rgi_area),length(time_cru)/12);
%V_error=nan(length(rgi_area),length(time_cru)/12);
%dL_cum_error=nan(length(rgi_area),length(time_cru)/12);
%mb_error=nan(length(rgi_area),length(time_cru)/12);
%dV_cum_error=nan(length(rgi_area),length(time_cru)/12-1);


load cru_grid lon_cru lat_cru elev_cru
load cru_clim
load cru_anom_406 

precip_cru_clim=precip_cru_clim.*precip_factor;

time_cru=(1901+(1/24)):(1/12):(2022-(1/24));

iteration_break_idx=[];

time(1)=cputime;

for i=1:length(rgi_area)

    if rgi_icecap(i)==1
        gamma=gamma_icecap;
        c_a=c_a_icecap;
        q=q_icecap;
        c_l=c_l_icecap;
   else
        gamma=gamma_glacier;
        c_a=c_a_glacier;
        q=q_glacier;
        c_l=c_l_glacier;
   end

    % find the lon and lat index of the glacier in the cru data 
    lon_cru_idx=min(find(min(abs(lon_cru(1,:)-rgi_lon(i)))==abs(lon_cru(1,:)-rgi_lon(i))));
    lat_cru_idx=min(find(min(abs(lat_cru(:,1)-rgi_lat(i)))==abs(lat_cru(:,1)-rgi_lat(i))));
    lon_cru_clim_idx=min(find(min(abs(lon_cru_clim(1,:)-rgi_lon(i)))==abs(lon_cru_clim(1,:)-rgi_lon(i))));
    lat_cru_clim_idx=min(find(min(abs(lat_cru_clim(:,1)-rgi_lat(i)))==abs(lat_cru_clim(:,1)-rgi_lat(i))));

    % extract temp, precip, and height from cru at the glacier location
    temp_cru_glacier=double(squeeze(temp_cru_anom(lon_cru_idx,lat_cru_idx,:)))+squeeze(repmat(squeeze(temp_cru_clim(lon_cru_clim_idx,lat_cru_clim_idx,:)),[length(time_cru)/12 1]));
    precip_cru_glacier=double(squeeze(precip_cru_anom(lon_cru_idx,lat_cru_idx,:)))+squeeze(repmat(squeeze(precip_cru_clim(lon_cru_clim_idx,lat_cru_clim_idx,:)),[length(time_cru)/12 1]));
    height_cru_glacier(i)=squeeze(elev_cru(lon_cru_idx,lat_cru_idx));
    temp_cru_clim_glacier=squeeze(temp_cru_clim(lon_cru_clim_idx,lat_cru_clim_idx,:));
    height_cru_clim_glacier(i)=squeeze(elev_cru_clim(lon_cru_clim_idx,lat_cru_clim_idx));

    if isnan(mean(precip_cru_glacier))
        nan_idx(i)=1;
    end

    % calculate lapse rates from surrounding points
    grid_dist=1;    
    %elev_surround=squeeze(elev_cru_clim(lon_cru_clim_idx-grid_dist:1:lon_cru_clim_idx+grid_dist,lat_cru_clim_idx-grid_dist:1:lat_cru_clim_idx+grid_dist));
    elev_surround=squeeze(elev_cru_clim(max(lon_cru_clim_idx-grid_dist,1):1:min(lon_cru_clim_idx+grid_dist,2160),max(lat_cru_clim_idx-grid_dist,1):1:min(lat_cru_clim_idx+grid_dist,856)));
    elev_surround=elev_surround(:); elev_surround(isnan(elev_surround))=[];
    %temp_clim_surround=mean(temp_cru_clim(lon_cru_clim_idx-grid_dist:1:lon_cru_clim_idx+grid_dist,lat_cru_clim_idx-grid_dist:1:lat_cru_clim_idx+grid_dist,:),3);
    temp_clim_surround=mean(temp_cru_clim(max(lon_cru_clim_idx-grid_dist,1):1:min(lon_cru_clim_idx+grid_dist,2160),max(lat_cru_clim_idx-grid_dist,1):1:min(lat_cru_clim_idx+grid_dist,856),:),3);
    temp_clim_surround=temp_clim_surround(:); temp_clim_surround(isnan(temp_clim_surround))=[];
    if length(elev_surround)<2
        elev_surround=squeeze(elev_cru_clim(max(1,lon_cru_clim_idx-(grid_dist+1)):1:min(2160,lon_cru_clim_idx+(grid_dist+1)),max(1,lat_cru_clim_idx-(grid_dist+1)):1:min(856,lat_cru_clim_idx+(grid_dist+1))));
        %elev_surround=squeeze(elev_cru_clim(lon_cru_clim_idx-(grid_dist+1):1:lon_cru_clim_idx+(grid_dist+1),lat_cru_clim_idx-(grid_dist+1):1:lat_cru_clim_idx+(grid_dist+1)));
        elev_surround=elev_surround(:); elev_surround(isnan(elev_surround))=[];
        temp_clim_surround=mean(temp_cru_clim(max(1,lon_cru_clim_idx-(grid_dist+1)):1:min(2160,lon_cru_clim_idx+(grid_dist+1)),max(1,lat_cru_clim_idx-(grid_dist+1)):1:min(856,lat_cru_clim_idx+(grid_dist+1)),:),3);
        %temp_clim_surround=mean(temp_cru_clim(lon_cru_clim_idx-(grid_dist+1):1:lon_cru_clim_idx+(grid_dist+1),lat_cru_clim_idx-(grid_dist+1):1:lat_cru_clim_idx+(grid_dist+1),:),3);
        temp_clim_surround=temp_clim_surround(:); temp_clim_surround(isnan(temp_clim_surround))=[];
    end
    if length(elev_surround)<2
        lapse_temp(i)=0.0065;
        rho_lapse_temp(i)=nan;
        p_lapse_temp(i)=nan;
    else
        lapse_t=polyfit(elev_surround,temp_clim_surround,1);
        lapse_temp(i)=-1*lapse_t(1);
        [rho_lapse_temp(i),p_lapse_temp(i)]=corr(elev_surround,temp_clim_surround);
    end    

    % calculate temp over glacier tongue by assuming lapse-rate 
    temp_cru_glacier_tongue=temp_cru_glacier+lapse_temp(i)*(height_cru_clim_glacier(i)-rgi_zmin(i));
    precip_cru_glacier=precip_cru_glacier.*(1+lapse_precip*(height_cru_clim_glacier(i)-(rgi_zmin(i)+rgi_zmax(i))/2));

    % remove non-solid precipitation 
    temp_range(i)=(rgi_zmax(i)-rgi_zmin(i)).*lapse_temp(i);
    factor=-1/temp_range(i).*temp_cru_glacier_tongue+1+temp_prec_solid./temp_range(i);
    factor=min(1,factor);
    factor=max(0,factor);
    
    precip_cru_glacier_total=precip_cru_glacier;
    precip_cru_glacier=precip_cru_glacier.*factor;

    % calculate distance to mb measurements
    %for j=1:length(lon_mb)
    %    dist(i,j)=vdist(rgi_lat(i),rgi_lon(i),lat_mb(j),lon_mb(j));
    %end

    %[~,sort_idx]=sort(dist(i,:));
    %mean_dist(i)=mean(dist(i,sort_idx(1:number_closest_glaciers)));
    
    % weigh by distance
    %weight_dist=1./dist(i,sort_idx(1:number_closest_glaciers))*(1/sum(1./dist(i,sort_idx(1:number_closest_glaciers))));
    
    % determine t_star
    %t_star_rgi(i)=round(sum(t_star(sort_idx(1:number_closest_glaciers)).*transpose(weight_dist)));
    %t_star_rgi(i)=t_star_cross(1);
    %t_star_rgi_idx=find(cru_years==t_star_rgi(i));
    
    %model_bias_rgi(i)=nansum(model_bias(t_star_rgi_idx,sort_idx(1:number_closest_glaciers)).*weight_dist);    

    % fill up temp and precip data with nans to allow calculation of
    % climatologies
    %temp_filled_nan=[nan(12*(length_of_climatology-1)/2,1);temp_cru_glacier_tongue;nan(12*(length_of_climatology-1)/2,1)];
    %precip_filled_nan=[nan(12*(length_of_climatology-1)/2,1);precip_cru_glacier;nan(12*(length_of_climatology-1)/2,1)];
        
    % select data from time window to match period of Hugonnet observations
    temp=temp_cru_glacier_tongue(find(abs(time_cru-(2000+1/24))<0.01):find(abs(time_cru-(2020-1/24))<0.01));
    precip=precip_cru_glacier(find(abs(time_cru-(2000+1/24))<0.01):find(abs(time_cru-(2020-1/24))<0.01));
                
    % calculate climatologies
    for k=1:12
    %    temp_glacier_tongue_clim(k)=nanmean(temp(k:12:length(temp)));
        precip_glacier_clim(k)=nanmean(precip(k:12:length(precip)));
    end
        
    % calculate model parameter
    mu_rgi(i)=(sum(precip)-20*1000*mb_hugonnet(i))/sum(max(0,temp-temp_melt));
    if isnan(mu_rgi(i)) | mu_rgi(i)<0
       nan_idx(i)=1;
    end

    turnover(i)=max(10,sum(precip_glacier_clim));

    V_pre=[];
    dL_pre=[];
    L_pre=[];
    A_pre_1=[];
    mb_modeled_pre=[];
    %sust_water_delay_pre=[];
    %mmp_sust_pre=[];
    %mmp_total_pre=[];
    mb_monthly_sust_pre=[];
    mb_monthly_total_pre=[];
    n_months_above_freezing_pre=[];

    % start modeling
    A_pre=rgi_area(i);

    for k=1:50 % start loop iteration

        V_pre=c_a*(A_pre(1).^gamma);
        dL_pre=0;
        L_pre=(V_pre(1)./c_l).^(1/q);
        A_pre_1(k)=A_pre(1);
        
        for j=1:rgi_year(i)-min(cru_years)

            mb_months=nan(1,12);
            months_idx=nan(1,12);
        
            if rgi_lat(i) >= 0
                mb_months=[-3/12+1/24:1/12:8/12+1/24]+double(cru_years(j)+1);
            elseif rgi_lat(i) < 0
                mb_months=[-9/12+1/24:1/12:2/12+1/24]+double(cru_years(j)+1);
            end
        
            for l=1:12
                months_idx(l)=find(abs(time_cru-mb_months(l))<1/1000);
            end
        
            delta_z=((rgi_zmax(i)-rgi_zmin(i))/((V_pre(1)./c_l).^(1/q)))*(((V_pre(1)./c_l).^(1/q))-L_pre(j));
            delta_z=max(delta_z,-rgi_zmin(i));

            delta_T_z=-delta_z*lapse_temp(i);
            
            precip=precip_cru_glacier(months_idx);
            temp=temp_cru_glacier_tongue(months_idx);
            
            mb_modeled_pre(j)=sum(precip)-mu_rgi(i)*(sum(max(0,delta_T_z+temp-temp_melt)));

            n_months_above_freezing_pre(j)=length(find(delta_T_z+temp-temp_melt>0)); 
           
            V_pre(j+1)=max(0,V_pre(j)+mb_modeled_pre(j)*A_pre(j)*1e-6);
 
            tau_L=max(1,V_pre(j)./(turnover(i).*A_pre(j).*1e-6));
            tau_A=max(1,tau_L.*(c_l.^(2/q))./(c_a^(1/gamma)).*(V_pre(j)^(1/gamma-2/q)));

            L_pre(j+1)=max(0,L_pre(j)+1/tau_L*(((V_pre(j+1)./c_l).^(1/q))-L_pre(j)));
            dL_pre(j+1)=1/tau_L*(((V_pre(j+1)./c_l).^(1/q))-L_pre(j));
            
            
            A_pre(j+1)=max(0,A_pre(j)+1/tau_A*(((V_pre(j+1)./c_a).^(1/gamma))-A_pre(j)));
            if V_pre(j+1)==0;
                L_pre(j+1)=0;
                dL_pre(j+1)=0;
                A_pre(j+1)=0;
            end
            
            
        end % end modeling 
        
	    A_pre_test=A_pre(cru_years==rgi_year(i));
        
        if abs(A_pre_test/rgi_area(i)-1)<0.01, break, end
        
        if A_pre_test<rgi_area(i) && k==1
            A_pre(1)=A_pre_1(k)*2;
        elseif A_pre_test>rgi_area(i) && k==1
            A_pre(1)=A_pre_1(k)*1/2;
        elseif A_pre_test<rgi_area(i) && k~=1
            if A_pre_1(k)==max(A_pre_1(1:k))
                A_pre(1)=A_pre_1(k)+rgi_area(i);
            else
                A_pre(1)=A_pre_1(k)+abs(A_pre_1(k)-A_pre_1(k-1))/2;
            end
        elseif A_pre_test>rgi_area(i) && k~=1
            A_pre(1)=A_pre_1(k)-abs(A_pre_1(k)-A_pre_1(k-1))/2;
        end        
        
    end % loop iteration

    if k==50
        iteration_break_idx=[iteration_break_idx i];
        nan_idx(i)=1;
    end

    V_pre=c_a*(A_pre(1).^gamma);
    dL_pre=0;
    L_pre=(V_pre(1)./c_l).^(1/q);
    A_pre_1(k)=A_pre(1);

    for j=1:length(cru_years)-1

            mb_months=nan(1,12);
            months_idx=nan(1,12);
        
            if rgi_lat(i) >= 0
                mb_months=[-3/12+1/24:1/12:8/12+1/24]+double(cru_years(j)+1);
            elseif rgi_lat(i) < 0
                mb_months=[-9/12+1/24:1/12:2/12+1/24]+double(cru_years(j)+1);
            end
        
            for l=1:12
                months_idx(l)=find(abs(time_cru-mb_months(l))<1/1000);
            end
        
            delta_z=((rgi_zmax(i)-rgi_zmin(i))/((V_pre(1)./c_l).^(1/q)))*(((V_pre(1)./c_l).^(1/q))-L_pre(j));
            delta_z=max(delta_z,-rgi_zmin(i));

            delta_T_z=-delta_z*lapse_temp(i);
            
            precip=precip_cru_glacier(months_idx);
            precip_total=precip_cru_glacier_total(months_idx);
            temp=temp_cru_glacier_tongue(months_idx);
            
            mb_modeled_pre(j)=sum(precip)-mu_rgi(i)*(sum(max(0,delta_T_z+temp-temp_melt)));

            mu_sust=[];
            mu_sust=sum(precip./12)./sum(max(0,delta_T_z+temp-temp_melt));

%            sust_water_delay_pre(j)=sum(max(mu_sust*(max(0,delta_T_z+temp-temp_melt))-precip,0));
%            mmp_sust_pre(j,:)=(max(mu_sust*(max(0,delta_T_z+temp-temp_melt))-precip,0));
%            mmp_total_pre(j,:)=(max(mu_rgi(i)*(max(0,delta_T_z+temp-temp_melt))-precip,0));
%            sust_water_delay_pre(j)=sum(max(mu_sust*(max(0,delta_T_z+temp-temp_melt))-(precip-(model_bias_rgi(i)+res_bias)./12),0));
%            mmp_sust_pre(j,:)=(max(mu_sust*(max(0,delta_T_z+temp-temp_melt))-(precip-(model_bias_rgi(i)+res_bias)./12),0));
%            mmp_total_pre(j,:)=(max(mu_rgi(i)*(max(0,delta_T_z+temp-temp_melt))-(precip-(model_bias_rgi(i)+res_bias)./12),0));
            mb_monthly_sust_pre(j,:)=(mu_sust*(max(0,delta_T_z+temp-temp_melt))-(precip./12));
            mb_monthly_total_pre(j,:)=(mu_rgi(i)*(max(0,delta_T_z+temp-temp_melt))-(precip./12));

            n_months_above_freezing_pre(j)=length(find(delta_T_z+temp-temp_melt>0)); 
           
            V_pre(j+1)=max(0,V_pre(j)+mb_modeled_pre(j)*A_pre(j)*1e-6);
 
            tau_L=max(1,V_pre(j)./(turnover(i).*A_pre(j).*1e-6));
            tau_A=max(1,tau_L.*(c_l.^(2/q))./(c_a^(1/gamma)).*(V_pre(j)^(1/gamma-2/q)));

            L_pre(j+1)=max(0,L_pre(j)+1/tau_L*(((V_pre(j+1)./c_l).^(1/q))-L_pre(j)));
            dL_pre(j+1)=1/tau_L*(((V_pre(j+1)./c_l).^(1/q))-L_pre(j));
            
            
            A_pre(j+1)=max(0,A_pre(j)+1/tau_A*(((V_pre(j+1)./c_a).^(1/gamma))-A_pre(j)));
            if V_pre(j+1)==0;
                L_pre(j+1)=0;
                dL_pre(j+1)=0;
                A_pre(j+1)=0;
            end
    end

    V(i,:)=V_pre;
    L(i,:)=L_pre;
    dL(i,:)=dL_pre;
    A(i,:)=A_pre;
    mb_modeled(i,:)=mb_modeled_pre;
%    sust_water_delay(i,:)=sust_water_delay_pre;
%    mmp_sust(i,:,:)=mmp_sust_pre;
%    mmp_total(i,:,:)=mmp_total_pre;
    mb_monthly_sust(i,:,:)=-mb_monthly_sust_pre;
    mb_monthly_total(i,:,:)=-mb_monthly_total_pre;
    dV(i,:)=mb_modeled_pre.*A_pre(1:length(cru_years)-1).*1e-6;
    
    n_months_above_freezing_pre(length(cru_years))=n_months_above_freezing_pre(length(cru_years)-1);
    n_months_above_freezing(i,:)=n_months_above_freezing_pre;


%     % calculate errors
%     
%     A_error(i,cru_years==rgi_year(i))=rgi_area(i)*area_error_rgi/100;
%     mb_error(i,cru_years==rgi_year(i))=mean(mse_mu_star_cross).^(1/2);
%     V_error(i,cru_years==rgi_year(i))=V(i,cru_years==rgi_year(i))*A_V_scaling_error;
%     L_error(i,cru_years==rgi_year(i))=L(i,cru_years==rgi_year(i))*L_V_scaling_error;
% 
%     dV_cum_error(i,cru_years==rgi_year(i))=min(V(i,cru_years==rgi_year(i)),abs(mb_modeled(i,cru_years==rgi_year(i))*A(i,cru_years==rgi_year(i))*1e-6))*((mb_error(i,cru_years==rgi_year(i))/max(10,abs(mb_modeled(i,cru_years==rgi_year(i)))))^2+(A_error(i,cru_years==rgi_year(i))/max(1e-6,A(i,cru_years==rgi_year(i))))^2)^(1/2);
%     dL_cum_error(i,cru_years==rgi_year(i))=abs(dL(i,cru_years==rgi_year(i))*((tau_L_error)^2+(L_V_scaling_error)^2)^(1/2));
%     start_j_idx=find(cru_years==rgi_year(i));
%         
%     for j=start_j_idx:length(cru_years)-1
%         dz=((rgi_zmax(i)-rgi_zmin(i))/((V(i,1)./c_l).^(1/q)))*(L(i,start_j_idx)-L(i,j+1));
%         dTz_error=lapse_temp(i)*dz*L_error(i,j)/max(1e-3,L(i,j));
%         mb_error(i,j+1)=(mean(mse_mu_star_cross)+n_months_above_freezing(i,j+1)*(mu_rgi(i)*dTz_error)^2)^(1/2);
%         
%         dV_error=min(V(i,j),abs(mb_modeled(i,j)*A(i,j)*1e-6))*((mb_error(i,j)/max(10,abs(mb_modeled(i,j))))^2+(A_error(i,j)/max(1e-6,A(i,j)))^2)^(1/2);
%         V_error(i,j+1)=(dV_error^2+V_error(i,j)^2)^(1/2);
%         
%         L_relax=(V(i,j+1)/c_l)^(1/q);
%         dL_error=abs(dL(i,j)*((tau_L_error)^2+(L_V_scaling_error)^2)^(1/2));
%         dL_cum_error(i,j+1)=((dL_cum_error(i,j))^2+dL_error^2)^(1/2);
%         L_error(i,j+1)=((L_error(i,j))^2+(dL_error)^2)^(1/2);
%         
%         A_relax=(V(i,j+1)/c_a)^(1/gamma);
%         dA=1/tau_A*(A_relax-A(i,j));
%         dA_error=abs(dA*((tau_A_error)^2+(A_V_scaling_error)^2)^(1/2));
%         A_error(i,j+1)=((A_error(i,j))^2+(dA_error)^2)^(1/2);
%         
%         if V(i,j+1)==0;
%             L_error(i,j+1)=L_error(i,j);
%             A_error(i,j+1)=A_error(i,j);
%         end
%     end
% 
%     for j=start_j_idx:length(cru_years)-2
%         dV_cum_error(i,j+1)=((dV_cum_error(i,j))^2+(min(V(i,j+1),abs(mb_modeled(i,j+1)*A(i,j+1)*1e-6))*((mb_error(i,j+1)/max(10,abs(mb_modeled(i,j+1))))^2+(A_error(i,j+1)/max(1e-6,A(i,j+1)))^2)^(1/2))^2)^(1/2);
%     end
%     
%     for j=start_j_idx-1:-1:1
%         dz=((rgi_zmax(i)-rgi_zmin(i))/((V(i,1)./c_l).^(1/q)))*(L(i,start_j_idx)-L(i,j));
%         dTz_error=lapse_temp(i)*dz*L_error(i,j+1)/max(1e-3,L(i,j+1));
%         mb_error(i,j)=(mean(mse_mu_star_cross)+n_months_above_freezing(i,j)*(mu_rgi(i)*dTz_error)^2)^(1/2);
%         
%         dV_error=min(V(i,j+1),abs(mb_modeled(i,j+1)*A(i,j+1)*1e-6))*((mb_error(i,j+1)/max(10,abs(mb_modeled(i,j+1))))^2+(A_error(i,j+1)/max(1e-6,A(i,j+1)))^2)^(1/2);
%         V_error(i,j)=(dV_error^2+V_error(i,j+1)^2)^(1/2);
%         
%         L_relax=(V(i,j)/c_l)^(1/q);
%         dL_error=abs(dL(i,j+1)*((tau_L_error)^2+(L_V_scaling_error)^2)^(1/2));
%         dL_cum_error(i,j)=((dL_cum_error(i,j+1))^2+dL_error^2)^(1/2);
%         L_error(i,j)=((L_error(i,j+1))^2+(dL_error)^2)^(1/2);
% 
%         A_relax=(V(i,j)/c_a)^(1/gamma);
%         dA=1/tau_A*(A_relax-A(i,j+1));
%         dA_error=abs(dA*((tau_A_error)^2+(A_V_scaling_error)^2)^(1/2));
%         A_error(i,j)=((A_error(i,j+1))^2+(dA_error)^2)^(1/2);
%         
%         dV_cum_error(i,j)=((dV_cum_error(i,j+1))^2+(min(V(i,j),abs(mb_modeled(i,j)*A(i,j)*1e-6))*((mb_error(i,j)/max(10,abs(mb_modeled(i,j))))^2+(A_error(i,j)/max(1e-6,A(i,j)))^2)^(1/2))^2)^(1/2);
% 
%     end
    
    time(2)=cputime;
    
    if i/50==round(i/50)
        sprintf('i = %d after %3.2f minutes, i.e. %d milliseconds per glacier \n%3.2f minutes to go',i,(time(2)-time(1))/60,round((time(2)-time(1))/i*1000),(length(rgi_area)-i)*((time(2)-time(1))/i)/60)
    end

    
end

height_cru_glacier(nan_idx==1)=[];
height_cru_clim_glacier(nan_idx==1)=[];
lapse_temp(nan_idx==1)=[];
rho_lapse_temp(nan_idx==1)=[];
p_lapse_temp(nan_idx==1)=[];
%mean_dist(nan_idx==1)=[];
%t_star_rgi(nan_idx==1)=[];
%model_bias_rgi(nan_idx==1)=[];
mu_rgi(nan_idx==1)=[];
turnover(nan_idx==1)=[];
A(nan_idx==1,:)=[];
L(nan_idx==1,:)=[];
V(nan_idx==1,:)=[];
dL(nan_idx==1,:)=[];
dV(nan_idx==1,:)=[];
mb_modeled(nan_idx==1,:)=[];
%sust_water_delay(nan_idx==1,:)=[];
%mmp_sust(nan_idx==1,:,:)=[];
%mmp_total(nan_idx==1,:,:)=[];
mb_monthly_sust(nan_idx==1,:,:)=[];
mb_monthly_total(nan_idx==1,:,:)=[];
n_months_above_freezing(nan_idx==1,:)=[];
% A_error(nan_idx==1,:)=[];
% L_error(nan_idx==1,:)=[];
% V_error(nan_idx==1,:)=[];
% dL_cum_error(nan_idx==1,:)=[];
% mb_error(nan_idx==1,:)=[];
% dV_cum_error(nan_idx==1,:)=[];
% %dist(:,nan_idx==1)=[];
% dist(nan_idx==1,:)=[];


eval(['save -v7.3 cru406_results_rgi_v6/results_region' region_number '.mat A L V dL dV height_cru_clim_glacier height_cru_glacier lapse_temp mb_modeled mb_monthly_sust mb_monthly_total mu_rgi n_months_above_freezing nan_idx p_lapse_temp rgi_* rho_lapse_temp turnover cru_years;'])

