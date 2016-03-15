'use strict';



$(document).ready(function () {
  // Show markdown format
  var md = new Remarkable('full');
  var markDIVs = $(".markdown");

  for(var i = 0; i < markDIVs.length; i++) {
    var markText = $(markDIVs[i]).html();
    var htmlText = md.render(markText);
    $(markDIVs[i]).html(htmlText);
  }

  // Show build-in math latex
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });

  // Build up kernel with control button
  var kernelDIVs = $(".kernel pre:first-child");
  for(var i = 0; i < kernelDIVs.length; i++) {
    var editorID = "code_"+i;
    $(kernelDIVs[i]).attr("id", editorID);
    var lines = $(kernelDIVs[i]).html().split(/\r\n|\r|\n/).length;
    var height = 14*lines;
    if ( height > 560) {
      height = 560;
    }
    $(kernelDIVs[i]).height(height);


    var kernel = $(kernelDIVs[i]).parent();

    var editor = ace.edit(editorID);
    editor.setTheme("ace/theme/xcode");
    editor.session.setMode("ace/mode/javascript");
    editor.renderer.setScrollMargin(10, 10);

    buildKernel(kernel, editor);
  }

  gui.init();
});

var buildKernel = function(kernelDIV, editor) {
    var buttonString = '<button type="button" class="btn btn-default btn-sm">'
                    +  '<span class="glyphicon glyphicon-play">'
                    +  '</span>Run</button> <span>Ready</span>';
    kernelDIV.append(buttonString);

    kernelDIV.children('button').bind('click',function(){
      kernelDIV.children('span').html("Running");
      try {
        jQuery.globalEval( editor.getValue() );
      } catch (e) {
        kernelDIV.children('span').html(e.message);
        return;
      }

      kernelDIV.children('span').html("Done");
    });
};
