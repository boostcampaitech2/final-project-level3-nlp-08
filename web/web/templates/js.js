function getParam(sname) {
    var params = location.search.substr(location.search.indexOf("?") + 1);
    var sval = "";

    params = params.split("&");

    for (var i = 0; i < params.length; i++) {
        temp = params[i].split("=");
        if ([temp[0]] == sname) { sval = temp[1]; }
    }
    return sval;
}


function removePoemCards() {
    document.getElementById('poem container').remove()
}

function createCardWithImage(poem) {
    var cardDiv = document.getElementById("poem-card-div");

    var newDIV = document.createElement("div");
    newDIV.setAttribute("class", "ui centered card");
    newDIV.setAttribute("id", "poem-card");
    newDIV.setAttribute("style", "width: 400px");

    var imgDIV = document.createElement("div");
    imgDIV.setAttribute("class", "image");

    var img = document.createElement("img");
    img.setAttribute("src", "web/assets/uploads/" + getParam('filename'));
    img.setAttribute("width", 400);
    img.setAttribute("height", 400);

    var contentDIV = document.createElement("div");
    contentDIV.setAttribute("class", "content");

    var contentDiscriptionDIV = document.createElement("div");
    contentDiscriptionDIV.setAttribute("class", "div");

    let innerPoem = ''
    for (const row of poem.split('\n')) {
        innerPoem += row + "<br>";
    }
    contentDiscriptionDIV.innerHTML = innerPoem;

    removePoemCards();
    contentDIV.appendChild(contentDiscriptionDIV);
    imgDIV.appendChild(img);

    newDIV.appendChild(imgDIV);
    newDIV.appendChild(contentDIV);

    let cardMsg = document.createElement('h2');
    cardMsg.innerHTML = 'Your Poem Card';
    cardMsg.setAttribute("class", "ui center aligned header");

    cardDiv.appendChild(cardMsg);
    cardDiv.appendChild(newDIV);

    let downloadButton = document.createElement('button');
    downloadButton.setAttribute('align', 'center');
    downloadButton.setAttribute('class', 'positive ui button')
    downloadButton.setAttribute('id', 'download');
    downloadButton.innerHTML = 'Download Card';
    downloadButton.setAttribute('onclick', 'download();');
    cardDiv.appendChild(downloadButton)

}

function download() {
    // 캡처 라이브러리를 통해 canvas 오브젝트 받고 이미지 파일로 리턴함
    html2canvas(document.querySelector("#poem-card")).then(canvas => {
        saveAs(canvas.toDataURL('image/jpg'), "poem_card.jpg"); //다운로드 되는 이미지 파일 이름 지정
    });
};
function saveAs(uri, filename) {
    // 캡처된 파일을 이미지 파일로 내보냄
    var link = document.createElement('a');
    if (typeof link.download === 'string') {
        link.href = uri;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } else {
        window.open(uri);
    }
};

{/* <button align="center" class="positive ui button" id="download">Download</button> */ }
{/* <h1 class="ui center aligned header">Image2Poem</h1>
<h3 class="ui center aligned header">Generate poem from image with AI</h3> */}


{/* <div class="ui card">
                <div class="image">
                    <img src="{{ url_for('display_image', filename=filename) }}" width="224" height="224">
                </div>
                <div class="content">
                    <div class="header"></div>
                    <div class="description">
                        {% for row in generated_poem.split('\n') %}
                        {{row}}<br>
                        {% endfor %}
                    </div>
                </div> */}