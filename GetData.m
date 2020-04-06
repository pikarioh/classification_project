function [data,identifier] = GetData()
%GETDATA Summary: Loads the .dat file and outputs the data table along with
%rowheader as identifier

data = textscan('vowdata_nohead.dat','U16');
identifier = data(0:1668,0);
data = data(0:1668,7:16);
end

