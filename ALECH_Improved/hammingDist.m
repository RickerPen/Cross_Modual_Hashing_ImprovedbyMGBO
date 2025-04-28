function DHamm = hammingDist(B2, B1)
    % 检查矩阵的列数是否相同
    [~, colB2] = size(B2);
    [~, colB1] = size(B1);
    if colB2 ~= colB1
        error('输入矩阵的列数必须相同');
    end
    
    % 获取矩阵的行数
    rowB2 = size(B2, 1);
    rowB1 = size(B1, 1);
    
    % 初始化汉明距离矩阵
    DHamm = zeros(rowB2, rowB1);
    
    % 计算汉明距离
    for i = 1:rowB2
        for j = 1:rowB1
            DHamm(i, j) = sum(B2(i, :) ~= B1(j, :));
        end
    end
end