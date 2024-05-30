function loginExampleAction() {
    var studentCode = $("input[name=studentCode]").val();
    var password = $("input[name=studentPassword]").val();
    var subject = $("#subject").val();
    var chapter = $("#chapter").val();
    if (!studentCode) {
        alert("请输入学号");
        return;
    }
    if (!password) {
        alert("请输入密码");
        return;
    }

    if (!subject) {
        alert("请选择练习科目");
        return;
    }

    var practiceType = "";
    if (chapter) {
        practiceType = "2";
    } else {
        practiceType = "1";
    }

    var params = {
        branchId: branchId,
        studentNum: studentCode,
        practiceType: practiceType,
        password: password,
    };
    $(".loadingMark").fadeIn();
    $.ajax({
        type: "post",
        url: "http://222.73.57.153:6571/login/practiceLogin",
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
            var subjectName = "";
            var chapterName = "";
            for (var i = 0; i < subjectList.length; i++) {
                var subjectItem = subjectList[i];
                if (subjectItem.value === subject) {
                    subjectName = subjectItem.label;
                }
            }
            for (var i = 0; i < chapterList.length; i++) {
                var chapterItem = chapterList[i];
                if (chapterItem.value === chapter) {
                    chapterName = chapterItem.label;
                }
            }

            var practice_userInfo = res.data.studentInfo;
            practice_userInfo.subjectName = subjectName;
            practice_userInfo.chapterName = chapterName;
            practice_userInfo.subjectId = subject;
            practice_userInfo.chapterId = chapter;
            practice_userInfo.branchId = branchId;
            localStorage.setItem(
                "practice_userInfo",
                JSON.stringify(practice_userInfo)
            );

            if (practiceType == 1) {
                window.location.href = "practiceSubject.html";
            } else {
                window.location.href = "practiceChapter.html";
            }
        },
        error: function (err) {
            $(".loadingMark").fadeOut();
        },
    });
}
