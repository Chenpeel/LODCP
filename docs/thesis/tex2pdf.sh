#!/bin/bash
# 论文编译与清理脚本

# 默认模式为编译
clean_mode=false

# 检查是否有参数
if [ "$1" = "-c" ]; then
    clean_mode=true
fi

# 清理功能
clean_files() {
    echo "清理临时文件..."
    find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" \
    -o -name "*.lof" -o -name "*.lot" -o -name "*.bbl" -o -name "*.bcf" \
    -o -name "*.blg" -o -name "*.run.xml" -o -name "*.synctex.gz" -o -name "*.xdv" \
    -o -name "*.fdb_latexmk" -o -name "*.fls" \) -delete
    echo "清理完成！"
}

# 编译功能
compile_thesis() {
    echo "第一次运行 xelatex..."
    xelatex thesis.tex

    echo "运行 biber 处理参考文献..."
    bibtex thesis

    echo "第二次运行 xelatex..."
    xelatex thesis.tex

    echo "第三次运行 xelatex（确保交叉引用正确）..."
    xelatex thesis.tex

    echo "编译完成！"
}

# 主逻辑
if $clean_mode; then
    clean_files
else
    compile_thesis
fi

# 退出
exit 0
