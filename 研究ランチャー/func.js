
function sT(){
  target = document.getElementById("id_start");
  const sTime = new Date();
  target.innerHTML = sTime.toLocaleString();
}

function fT(){
  target_start = document.getElementById("id_start");
  target_finish = document.getElementById("id_finish");
  target_spend = document.getElementById("id_spend");
  const fTime = new Date();
  const sTime = new Date(target_start.innerHTML);

  target_finish.innerHTML = fTime.toLocaleString();
  spend = calcSpend(sTime, fTime);
  hour = spend[0]; minute = spend[1]; second = spend[2];

  target_spend.innerHTML = hour + ' 時間 ' + minute + ' 分 ' + second + ' 秒';
  // save(fTime, spend);
}

function calcSpend(sTime, fTime){
  mSecond = fTime.getTime() - sTime.getTime();
  second = mSecond / 1000;


  minute = second / 60;
  second = second % 60;

  hour = minute / 60;
  minute % 60;

  second = parseInt(second);
  minute = parseInt(minute);
  hour = parseInt(hour);

  return [hour, minute, second];
}
//
// function save(fTime, spend){
//     // target_spend.innerHTML = fTime.getFullYear()
//     var json = [
//       "year" : "2020",
//       "month" : "11",
//       "day" : "3",
//       "hour" : "0",
//       "minute" : "2",
//       "second" : "20",
//     ];
//     // 保存するJSONファイルの名前
//     const fileName = "date.json";
//     // データをJSON形式の文字列に変換する。
//     const data = JSON.stringify(json);
//     // HTMLのリンク要素を生成する。
//     const link = document.createElement("a");
//     // リンク先にJSON形式の文字列データを置いておく。
//     link.href = "data:text/plain," + encodeURIComponent(data);
//     // 保存するJSONファイルの名前をリンクに設定する。
//     a.download = fileName;
//     // ファイルを保存する。
//     a.click();
// }


function save(){
  var fs = require('fs');

var text = "hoge foo bar";
fs.writeFile('hoge.txt', text);


}
