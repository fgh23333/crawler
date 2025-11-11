function getData() {
    var info = JSON.parse(localStorage.getItem("practice_userInfo"));
    var params = {
        subjectId: info.subjectId,
        chapterId: info.chapterId,
        branchId: info.branchId,
        studentId: info.id,
    };
    $(".loadingMark").fadeIn();
    $.ajax({
        type: "post",
        url: "http://222.73.57.153:6571/examinationInfo/getPracticeInfo",
        dataType: "json",
        cache: false,
        contentType: "application/json;charset=utf-8",
        data: JSON.stringify(params),
        success: function (res) {
            $(".loadingMark").fadeOut();
            if (parseInt(res.code) != 200) {
                alert(res.msg);
                return;
            }
            $(".nextBtn").css("display", "none");
            $(".backBtn").css("display", "none");
            $(".againBtn").css("display", "none");
            $(".saveBtn").css("display", "none");
            data = res.data.topicInfo;
            data.title = data.topicContent;
            data.options = data.topicSelect;
            data.standardAnswer = data.yesAnswer;
            updateExamData();
        },
        error: function (err) {
            $(".loadingMark").fadeOut();
        },
    });
}

function saveQues() {
    if (!data) {
        return;
    }
    let tip = [];
    if (!data.answer) {
        alert("请先答题");
        return;
    }
    hasCommit = false;
    $(".nextBtn").css("display", "none");
    $(".backBtn").css("display", "none");
    loadChapterQuestion();
    if (data.answer === data.standardAnswer) {

        $(".saveBtn").css("display", "none");
        $(".againBtn").css("display", "none");
        alert("回答正确")
        getData()
    } else {
        alert("回答错误")
    }
}

saveQues()