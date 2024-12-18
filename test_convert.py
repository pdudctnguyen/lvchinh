def extract_text_by_line(ocr_data, y_threshold=15):
    from collections import defaultdict
    
    lines = defaultdict(list)

    # Gom nhóm theo dòng dựa trên tọa độ y (với khoảng sai số)
    for box, text in ocr_data:
        y_center = (box[0][1] + box[2][1]) // 2
        added = False
        
        # Tìm dòng gần nhất để thêm đoạn vào
        for y in lines:
            if abs(y_center - y) <= y_threshold:
                lines[y].append((box, text))
                added = True
                break
        
        # Nếu không tìm thấy dòng gần, tạo dòng mới
        if not added:
            lines[y_center].append((box, text))
    
    # Sắp xếp các dòng theo y
    sorted_lines = sorted(lines.items())
    
    result_lines = []
    for y, segments in sorted_lines:
        # Sắp xếp các đoạn trên dòng theo x
        segments.sort(key=lambda seg: seg[0][0][0])
        line_text = ' '.join([seg[1] for seg in segments])
        result_lines.append(line_text.strip())  # Xóa khoảng trắng dư
    
    return '\n'.join(result_lines)

data = [
    ([[130, 24], [418, 24], [418, 54], [130, 54]], 'HỌC VIỆN CHÍNH TRỊ QUỐC GIA'),
([[458, 26], [740, 26], [740, 54], [458, 54]], 'ĐẢNG CỘNG SẢN VIỆT NAM'),
([[222, 50], [348, 50], [348, 74], [222, 74]], 'HỒ CHÍ MINH'),
([[112, 70], [438, 70], [438, 98], [112, 98]], 'HỌC VIỆN CHÍNH TRỊ KHU VỰC III'),
([[476.0513167019495, 117.15395010584845], [526.8624394627632, 126.10197977638802], [521.9486832980505, 152.84604989415155], [470.1375605372368, 143.89802022361198]], 'Nằng'),
([[174, 122], [202, 122], [202, 146], [174, 146]], 'Số'),
([[237, 122], [378, 122], [378, 148], [237, 148]], '-QĐ/HVCTKV III'),
([[582, 124], [724, 124], [724, 150], [582, 150]], 'tháng 4 năm 2022'),
([[447, 125], [475, 125], [475, 145], [447, 145]], 'Đà'),
([[522, 126], [566, 126], [566, 150], [522, 150]], 'ngày'),
([[348, 190], [484, 190], [484, 218], [348, 218]], 'QUYẾT ĐỊNH'),
([[118, 212], [716, 212], [716, 241], [118, 241]], 'Về việc cử cán bộ tham dự Lễ khai giáng và giảng lớp Bồi dưởng, cập nhật'),
([[194, 234], [640, 234], [640, 260], [194, 260]], 'kiến thức cho bí thư cấp ủy cấp xã, nhiệm kỳ 2020-2025,'),
([[334, 256], [498, 256], [498, 284], [334, 284]], 'tại tỉnh Quảng Bình'),
([[546.9580569325292, 303.1867537178289], [598.8734788556635, 311.13795634330097], [594.0419430674708, 335.8132462821711], [543.1265211443365, 327.86204365669903]], 'tháng'),
([[146, 306], [546, 306], [546, 334], [146, 334]], 'Căn cứ Quyết định số 6589-QĐ/HVCTQG ngày 01'),
([[622, 308], [734, 308], [734, 332], [622, 332]], 'năm 2018 của'),
([[98, 332], [734, 332], [734, 360], [98, 360]], 'Giám đốc Học viện Chính trị quốc gia Hồ Chí Minh về chức năng, nhiệm vụ, quyền'),
([[98, 358], [524, 358], [524, 384], [98, 384]], 'hạn, tổ chức bộ máy của Học viện Chính trị khu vực III;'),
([[238, 391], [734, 391], [734, 419], [238, 419]], 'định số 161/QyĐ-HVCT-HCKV III ngày 28 tháng 3 năm 2012 của'),
([[203.2025477771711, 392.1658598874935], [240.8534785364468, 397.24858978252155], [237.7974522228289, 418.8341401125065], [200.1465214635532, 413.75141021747845]], 'Quy'),
([[147, 395], [205, 395], [205, 415], [147, 415]], 'Căn cú'),
([[98, 418], [318, 418], [318, 442], [98, 442]], 'Giám đốc Học viện Chính trị'),
([[505.17239411139764, 419.17926704507477], [542.8534785364468, 423.24858978252155], [539.8276058886024, 444.82073295492523], [502.1465214635532, 439.75141021747845]], '(nay'),
([[542, 420], [734, 420], [734, 444], [542, 444]], 'là Học viện Chính trị khu'),
([[327, 421], [505, 421], [505, 441], [327, 441]], 'Hành chính khu vực II'),
([[307.2572186472918, 444.1430466182295], [348.86315523897014, 449.27291502869457], [345.7427813527082, 472.8569533817705], [304.13684476102986, 466.72708497130543]], 'công'),
([[344, 446], [378, 446], [378, 470], [344, 470]], 'tác;'),
([[101, 447], [307, 447], [307, 467], [101, 467]], 'Vực III) về việc cử cán bộ đi'),
([[146, 476], [432, 476], [432, 506], [146, 506]], 'Xét đề nghị của Trưởng Ban Tổ chức'),
([[441, 481], [499, 481], [499, 501], [441, 501]], 'Cán bộ'),
([[274, 516], [606, 516], [606, 546], [274, 546]], 'GIÁM ĐỐC HỌC VIỆN QUYẾT ĐỊNH'),
([[238.99099080900552, 552.1747742652154], [284.8266440708374, 559.9949709970356], [280.00900919099445, 585.8252257347846], [234.17335592916257, 578.0050290029644]], 'đồng'),
([[144, 556], [240, 556], [240, 580], [144, 580]], 'Điều 1.Cử'),
([[282, 556], [734, 556], [734, 584], [282, 584]], 'Chí A, Phó Giám đốc Học viện tham dự Lễ khai giảng và'),
([[96, 584], [734, 584], [734, 612], [96, 612]], 'giảng lớp Bồi dường, cập nhật kiến thức cho bí thư cấp ủy cấp xã, nhiệm kỳ 2020-'),
([[396.95805693252925, 609.1867537178289], [447.8685998665223, 617.1218571837177], [444.04194306747075, 641.8132462821711], [393.1314001334777, 633.8781428162823]], 'tháng'),
([[538.9580569325292, 609.1867537178289], [599.9075055932809, 617.2608036627252], [596.0419430674708, 641.8132462821711], [535.0924944067191, 633.7391963372748]], '(không'),
([[98, 613], [398, 613], [398, 640], [98, 640]], '2025 tại tỉnh Quảng Bình, vào ngày 18'),
([[594, 614], [734, 614], [734, 640], [594, 640]], 'kể thời gian đi và'),
([[447, 615], [539, 615], [539, 635], [447, 635]], '4 năm 2022'),
([[96, 640], [128, 640], [128, 664], [96, 664]], 'về)'),
([[214, 674], [548, 674], [548, 705], [214, 705]], 'Chánh Văn phòng, Trưởng Ban Tổ chức'),
([[562, 674], [734, 674], [734, 704], [562, 704]], 'Cán bộ, Trưởng Ban'),
([[144, 676], [208, 676], [208, 700], [144, 700]], 'Điều 2'),
([[522.9230627620734, 701.1999631813908], [579.8942914637132, 709.2106477826237], [576.0769372379266, 732.8000368186092], [519.1057085362868, 725.7893522173763]], 'trưởng'),
([[98, 704], [400, 704], [400, 730], [98, 730]], 'Quản lý đào tạo, Trưởng Ban Kế hoạch'),
([[408, 706], [524, 706], [524, 730], [408, 730]], 'Tài chính, Thủ'),
([[578, 706], [734, 706], [734, 730], [578, 730]], 'các đơn vị liên quan'),
([[120.05131670194949, 726.1539501058485], [166.8015702550885, 733.9270582001789], [160.94868329805053, 761.8460498941515], [114.19842974491151, 754.0729417998211]], 'đồng'),
([[204, 732], [412, 732], [412, 758], [204, 758]], 'căn cứ Quyết định thi hành'),
([[161, 735], [189, 735], [189, 755], [161, 755]], 'chí'),
([[105, 771], [185, 771], [185, 791], [105, 791]], 'Nơi nhận'),
([[518, 774], [624, 774], [624, 800], [518, 800]], 'GIÁM ĐỐC'),
([[117, 795], [197, 795], [197, 813], [117, 813]], 'Như Điều 2'),
([[115, 813], [231, 813], [231, 833], [115, 833]], 'Lưu: VT, TC-CB')
]
# Sắp xếp văn bản theo dòng (trục Y)
data.sort(key=lambda x: (x[0][0][1], x[0][0][0]))  # Sắp xếp theo Y trước, sau đó đến X

# Nhóm các đoạn văn bản trên cùng một dòng
lines = {}
for box, text in data:
    y_center = (box[0][1] + box[2][1]) // 2  # Tính trung điểm Y
    found_line = False
    for y in lines:
        if abs(y - y_center) < 20:  # Khoảng cách gần trên cùng một dòng
            lines[y].append((box[0][0], text))
            found_line = True
            break
    if not found_line:
        lines[y_center] = [(box[0][0], text)]

# Sắp xếp và kết nối các dòng văn bản
final_text = []
for y in sorted(lines.keys()):
    line_text = ' '.join(text for _, text in sorted(lines[y], key=lambda x: x[0]))
    final_text.append(line_text)

# Kết quả văn bản hoàn chỉnh
result = '\n'.join(final_text)
print(result)
# Chạy hàm xử lý
# formatted_text = extract_text_by_line(ocr_data)
# print(formatted_text)