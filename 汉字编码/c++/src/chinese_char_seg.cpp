#include<bits/stdc++.h>
using namespace std;
void utf8_split_character()
{
    int f_len,num=0,c_len;//文件字节数,character数量,单个字符长度
    ifstream f_in("../data/utf-8_input.txt",ios::in|ios::binary);//以二进制形式读文件
    f_in.seekg(0,ios::end);//把指针移到文件末尾
    f_len=f_in.tellg();//获取文件长度
    f_in.seekg(0,ios::beg);//把指针移到文件开头
    char *buffer=new char[f_len];//开辟文件长度大小的空间
    f_in.read(buffer,f_len);//按字节读文件,全部存到buffer中
    string s=buffer;//buffer的内容存到s,方便操作
    f_in.close();
    ofstream f_out("../result/utf-8_output.txt");//输出结果的文件
    for(int i=0; i<f_len;)
    {
        bitset<8> byte(s[i]);//单字节转二进制
        if(s[i]==' ')//遇到空格跳过(utf-8,gbk,ascii码空格均为0x20)
        {
            i=i+1;
            continue;
        }
        else if(byte[7]==0) c_len=1;//0xxxxxxx,1字节
        else if(byte[5]==0) c_len=2;//110xxxxx,2字节
        else if(byte[4]==0) c_len=3;//11110xxx,3字节
        else if(byte[3]==0) c_len=4;//111110xx,4字节
        else if(byte[2]==0) c_len=5;//1111110x,5字节
        else if(byte[1]==0) c_len=6;//11111110,6字节
        string c=s.substr(i,c_len);
        f_out<<c;
        if(i+c_len<f_len) f_out<<' ';//最后一个字符后面没有空格
        num++;
        i=i+c_len;
    }
    cout<<"utf-8编码分字结果："<<endl<<"请见文件：utf-8_output.txt"<<endl;
    cout<<"character数量："<<num<<endl<<endl;
    f_out.close();
}
void gbk_split_character()
{
    int f_len,num=0,c_len;//文件字节数,character数量,单个字符长度
    ifstream f_in("../data/gbk_input.txt",ios::in|ios::binary);//以二进制形式读文件
    f_in.seekg(0,ios::end);//把指针移到文件末尾
    f_len=f_in.tellg();//获取文件长度
    f_in.seekg(0,ios::beg);//把指针移到文件开头
    char *buffer=new char[f_len];//开辟文件长度大小的空间
    f_in.read(buffer,f_len);//按字节读文件,全部存到buffer中
    string s=buffer;//buffer的内容存到s,方便操作
    f_in.close();
    ofstream f_out("../result/gbk_output.txt");//输出结果的文件
    cout<<"gbk编码分字结果："<<endl;
    for(int i=0; i<f_len;)
    {
        bitset<8> byte(s[i]);//转二进制
        if(s[i]==' ')//遇到空格跳过(utf-8,gbk,ascii码空格均为0x20)
        {
            i=i+1;
            continue;
        }
        else if(byte[7]==0) c_len=1;//首位是0,1字节
        else if(byte[7]==1) c_len=2;//首位是1,2字节
        string c=s.substr(i,c_len);
        cout<<c;
        f_out<<c;
        if(i+c_len<f_len)//最后一个字符后面没有空格
        {
            cout<<' ';
            f_out<<' ';
        }
        num++;
        i=i+c_len;
    }
    cout<<endl<<"character数量："<<num<<endl;
    f_out<<endl<<"character数量："<<num<<endl;
    f_out.close();
}
int main()
{
    utf8_split_character();//utf-8编码分字
    gbk_split_character();//gbk编码分字
    return 0;
}
