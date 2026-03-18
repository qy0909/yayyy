"""Centralized multilingual k-shot examples for prompt construction."""

KSHOT_LIBRARY = {
    'jv': {
        'examples': [
            {
                'user': 'Aku kerja 12 jam saben dina, nanging lembur ora dibayar.',
                'detected_lang': 'Javanese',
                'assistant': 'Kondisi kuwi ora adil. Catet jam kerja saben dina, simpen bukti chat utawa slip gaji, banjur lapor menyang pejabat tenaga kerja.'
            },
            {
                'user': 'Pasporku dicekel bos, aku kudu piye?',
                'detected_lang': 'Javanese',
                'assistant': 'Paspor iku dokumen pribadi. Jaluk bali kanthi apik lan nek ora diwenehke, goleki bantuan pejabat tenaga kerja utawa polisi.'
            },
            {
                'user': 'Aku pengin ngerti cara gawe aduan resmi.',
                'detected_lang': 'Javanese',
                'assistant': 'Gawe aduan nganggo langkah cetha: kumpul bukti, tulis kronologi, terus kirim aduan ing pejabat tenaga kerja paling cedhak.'
            },
            {
                'user': 'Aku ora paham formulir online, ana cara liyane?',
                'detected_lang': 'Javanese',
                'assistant': 'Ana. Kowe iso langsung teka kantor layanan supaya petugas bantu isi formulir kanthi langsung.'
            },
            {
                'user': 'Gajiku telat rong wulan, apa sing kudu tak siapke?',
                'detected_lang': 'Javanese',
                'assistant': 'Siapke kontrak kerja, catatan jam kerja, slip gaji lawas, lan bukti transfer bank supaya proses aduan luwih kuat.'
            },
            {
                'user': 'Aku wedi nek bos mbales nek aku ngadu.',
                'detected_lang': 'Javanese',
                'assistant': 'Roso wedi kuwi wajar. Nanging kowe nduwe hak lapor, lan petugas biasane iso nerangake cara aman kanggo terus proses.'
            },
            {
                'user': 'Kantor tenaga kerja cedhak kene ana ngendi?',
                'detected_lang': 'Javanese',
                'assistant': 'Sebutno kutha utawa daerahmu, ben tak bantu golek alamat lan nomer telpon kantor tenaga kerja sing paling cedhak.'
            },
            {
                'user': 'Aku butuh versi ceklis sing gampang kanggo dibawa.',
                'detected_lang': 'Javanese',
                'assistant': 'Bisa. Tak gawe ceklis sederhana: dokumen identitas, kontrak, bukti gaji, catatan jam kerja, lan langkah lapor siji-siji.'
            }
        ]
    },
    'id': {
        'examples': [
            {
                'user': 'Saya kerja lembur hampir tiap hari tapi tidak dibayar.',
                'detected_lang': 'Indonesian',
                'assistant': 'Itu pelanggaran hak pekerja. Catat jam kerja Anda, kumpulkan bukti, lalu ajukan pengaduan ke kantor ketenagakerjaan terdekat.'
            },
            {
                'user': 'Paspor saya ditahan oleh majikan. Apa yang harus saya lakukan?',
                'detected_lang': 'Indonesian',
                'assistant': 'Paspor adalah dokumen pribadi. Minta dikembalikan secara baik, dan jika ditolak segera minta bantuan kantor ketenagakerjaan atau polisi.'
            },
            {
                'user': 'Bagaimana cara melapor gaji yang belum dibayar?',
                'detected_lang': 'Indonesian',
                'assistant': 'Mulai dengan tiga langkah: kumpulkan bukti gaji, tulis kronologi tanggal dan jumlah, lalu laporkan resmi ke dinas ketenagakerjaan.'
            },
            {
                'user': 'Saya tidak bisa isi form online, ada bantuan langsung?',
                'detected_lang': 'Indonesian',
                'assistant': 'Ada. Anda bisa datang langsung ke kantor layanan agar petugas membantu pengisian formulir secara tatap muka.'
            },
            {
                'user': 'Dokumen apa saja yang wajib dibawa saat aduan?',
                'detected_lang': 'Indonesian',
                'assistant': 'Bawa identitas, kontrak kerja, slip gaji, catatan jam kerja, dan bukti chat atau transfer agar proses verifikasi lebih cepat.'
            },
            {
                'user': 'Saya takut dibalas oleh majikan kalau melapor.',
                'detected_lang': 'Indonesian',
                'assistant': 'Rasa takut itu wajar. Anda tetap berhak melapor, dan petugas biasanya dapat memberi panduan langkah aman selama proses berlangsung.'
            },
            {
                'user': 'Bisa bantu cari kantor tenaga kerja terdekat di kota saya?',
                'detected_lang': 'Indonesian',
                'assistant': 'Bisa. Sebutkan nama kota atau wilayah Anda, nanti saya bantu berikan alamat dan kontak kantor yang paling dekat.'
            },
            {
                'user': 'Tolong buatkan ringkasan langkah yang simpel.',
                'detected_lang': 'Indonesian',
                'assistant': 'Ringkasnya: simpan semua bukti, catat kronologi, datang ke kantor tenaga kerja, lalu ikuti arahan petugas sampai kasus selesai.'
            }
        ]
    },
    'ms': {
        'examples': [
            {
                'user': 'Saya kerja lebih masa setiap minggu tapi bayaran lebih masa tiada.',
                'detected_lang': 'Malay',
                'assistant': 'Situasi itu tidak adil. Simpan rekod jam kerja dan bukti gaji, kemudian buat aduan di pejabat Jabatan Tenaga Kerja berdekatan.'
            },
            {
                'user': 'Majikan pegang pasport saya. Apa langkah selamat?',
                'detected_lang': 'Malay',
                'assistant': 'Pasport ialah dokumen peribadi anda. Minta semula secara baik dan dapatkan bantuan pihak berkuasa jika majikan enggan pulangkan.'
            },
            {
                'user': 'Bagaimana saya mahu lapor gaji tertunggak?',
                'detected_lang': 'Malay',
                'assistant': 'Mula dengan kumpul bukti, tulis kronologi, dan hantar aduan rasmi ke Jabatan Tenaga Kerja untuk siasatan lanjut.'
            },
            {
                'user': 'Saya kurang mahir guna borang online.',
                'detected_lang': 'Malay',
                'assistant': 'Tiada masalah. Anda boleh hadir ke kaunter fizikal supaya pegawai bantu isi borang secara terus.'
            },
            {
                'user': 'Dokumen apa perlu saya bawa semasa aduan?',
                'detected_lang': 'Malay',
                'assistant': 'Bawa kad pengenalan atau pasport, kontrak kerja, slip gaji, rekod waktu kerja, dan mesej berkaitan majikan.'
            },
            {
                'user': 'Saya risau majikan marah jika saya buat aduan.',
                'detected_lang': 'Malay',
                'assistant': 'Kerisauan itu normal. Anda tetap ada hak untuk membuat aduan, dan pegawai boleh terangkan langkah perlindungan yang sesuai.'
            },
            {
                'user': 'Boleh bantu saya cari pejabat tenaga kerja paling dekat?',
                'detected_lang': 'Malay',
                'assistant': 'Boleh. Beritahu negeri atau bandar anda dan saya akan beri alamat serta nombor telefon pejabat terdekat.'
            },
            {
                'user': 'Ringkaskan apa yang saya perlu buat sekarang.',
                'detected_lang': 'Malay',
                'assistant': 'Ringkas: simpan bukti, catat jumlah tuntutan, pergi ke pejabat tenaga kerja, dan ikut arahan pegawai untuk tindakan lanjut.'
            }
        ]
    },
    'tl': {
        'examples': [
            {
                'user': 'Pinagtatrabaho ako ng overtime pero walang bayad.',
                'detected_lang': 'Tagalog',
                'assistant': 'Hindi iyon tama. Itala ang oras ng trabaho mo, ipunin ang ebidensya, at magsampa ng reklamo sa pinakamalapit na labor office.'
            },
            {
                'user': 'Hawak ng employer ko ang passport ko. Ano ang gagawin ko?',
                'detected_lang': 'Tagalog',
                'assistant': 'Personal na dokumento ang passport mo. Hilingin itong ibalik at humingi ng tulong sa labor office o pulis kung tumanggi sila.'
            },
            {
                'user': 'Paano ako magrereklamo tungkol sa hindi nabayarang sahod?',
                'detected_lang': 'Tagalog',
                'assistant': 'Unahin mo ang ebidensya ng sahod, gumawa ng malinaw na timeline, at magsumite ng pormal na reklamo sa tamang tanggapan.'
            },
            {
                'user': 'Hindi ako marunong mag-fill up ng online form.',
                'detected_lang': 'Tagalog',
                'assistant': 'Pwede kang pumunta sa physical office para matulungan ka ng staff sa pag-fill up ng form.'
            },
            {
                'user': 'Anong dokumento ang dapat kong dalhin?',
                'detected_lang': 'Tagalog',
                'assistant': 'Dalhin ang ID o passport, kontrata, payslip, tala ng oras, at mga mensahe na may kinalaman sa employer.'
            },
            {
                'user': 'Natatakot ako na gantihan ako kapag nagreklamo.',
                'detected_lang': 'Tagalog',
                'assistant': 'Normal ang takot na iyan. May karapatan kang magreklamo at puwedeng magbigay ang opisyal ng gabay para mas ligtas ang proseso.'
            },
            {
                'user': 'Pwede mo ba akong tulungan hanapin ang pinakamalapit na labor office?',
                'detected_lang': 'Tagalog',
                'assistant': 'Oo. Sabihin mo lang ang lungsod o rehiyon mo para maibigay ko ang address at contact details ng pinakamalapit na opisina.'
            },
            {
                'user': 'Gusto ko ng maikling checklist na madaling sundan.',
                'detected_lang': 'Tagalog',
                'assistant': 'Narito ang simple checklist: ebidensya, listahan ng oras, dokumento ng pagkakakilanlan, at pagpunta sa labor office para magsumite.'
            }
        ]
    },
    'th': {
        'examples': [
            {
                'user': 'ฉันทำโอทีทุกวันแต่ไม่ได้ค่าล่วงเวลา',
                'detected_lang': 'Thai',
                'assistant': 'กรณีนี้ไม่เป็นธรรม ควรจดชั่วโมงทำงาน เก็บหลักฐานการจ่ายเงิน และยื่นเรื่องที่สำนักงานแรงงานใกล้บ้าน'
            },
            {
                'user': 'นายจ้างยึดพาสปอร์ตของฉันไว้ ต้องทำอย่างไร',
                'detected_lang': 'Thai',
                'assistant': 'พาสปอร์ตเป็นเอกสารส่วนตัว ควรขอคืนอย่างสุภาพ และหากไม่ได้คืนให้ติดต่อสำนักงานแรงงานหรือสถานีตำรวจ'
            },
            {
                'user': 'จะร้องเรียนเรื่องเงินเดือนไม่จ่ายได้อย่างไร',
                'detected_lang': 'Thai',
                'assistant': 'เริ่มจากรวบรวมหลักฐานเงินเดือน บันทึกเหตุการณ์ตามเวลา แล้วส่งคำร้องอย่างเป็นทางการกับหน่วยงานแรงงาน'
            },
            {
                'user': 'ฉันกรอกแบบฟอร์มออนไลน์ไม่เป็น',
                'detected_lang': 'Thai',
                'assistant': 'คุณสามารถไปที่สำนักงานบริการโดยตรง เพื่อให้เจ้าหน้าที่ช่วยกรอกแบบฟอร์มให้ได้'
            },
            {
                'user': 'ต้องเตรียมเอกสารอะไรไปยื่นเรื่อง',
                'detected_lang': 'Thai',
                'assistant': 'ควรเตรียมบัตรประจำตัวหรือพาสปอร์ต สัญญาจ้าง สลิปเงินเดือน บันทึกเวลา และข้อความที่เกี่ยวข้องกับนายจ้าง'
            },
            {
                'user': 'ฉันกลัวว่าจะโดนตอบโต้ถ้าไปแจ้งเรื่อง',
                'detected_lang': 'Thai',
                'assistant': 'ความกังวลนี้เป็นเรื่องปกติ คุณยังมีสิทธิร้องเรียน และเจ้าหน้าที่สามารถแนะนำวิธีดำเนินการอย่างปลอดภัย'
            },
            {
                'user': 'ช่วยหาสำนักงานแรงงานที่ใกล้ที่สุดให้หน่อย',
                'detected_lang': 'Thai',
                'assistant': 'ได้ครับ กรุณาบอกจังหวัดหรือเขตของคุณ แล้วฉันจะช่วยหาที่อยู่และเบอร์ติดต่อของสำนักงานที่ใกล้ที่สุด'
            },
            {
                'user': 'ขอสรุปเป็นเช็กลิสต์สั้นๆได้ไหม',
                'detected_lang': 'Thai',
                'assistant': 'สรุปสั้นๆคือ เก็บหลักฐาน บันทึกรายละเอียด เตรียมเอกสาร และไปยื่นเรื่องที่สำนักงานแรงงาน'
            }
        ]
    },
    'vi': {
        'examples': [
            {
                'user': 'Tôi làm tăng ca mỗi ngày nhưng không được trả tiền.',
                'detected_lang': 'Vietnamese',
                'assistant': 'Trường hợp này không công bằng. Bạn nên ghi lại giờ làm, lưu bằng chứng và nộp khiếu nại tại văn phòng lao động gần nhất.'
            },
            {
                'user': 'Chủ giữ hộ chiếu của tôi, tôi phải làm gì?',
                'detected_lang': 'Vietnamese',
                'assistant': 'Hộ chiếu là giấy tờ cá nhân của bạn. Hãy yêu cầu trả lại và liên hệ cơ quan lao động hoặc công an nếu bị từ chối.'
            },
            {
                'user': 'Làm sao để khiếu nại lương chưa trả?',
                'detected_lang': 'Vietnamese',
                'assistant': 'Bạn hãy chuẩn bị chứng cứ lương, viết rõ mốc thời gian, rồi nộp đơn khiếu nại chính thức tại cơ quan chức năng.'
            },
            {
                'user': 'Tôi không biết điền mẫu đơn online.',
                'detected_lang': 'Vietnamese',
                'assistant': 'Không sao, bạn có thể đến trực tiếp quầy dịch vụ để được nhân viên hỗ trợ điền biểu mẫu.'
            },
            {
                'user': 'Cần mang theo giấy tờ gì khi nộp đơn?',
                'detected_lang': 'Vietnamese',
                'assistant': 'Nên mang giấy tờ tùy thân, hợp đồng lao động, bảng lương, ghi chú giờ làm và tin nhắn liên quan với chủ sử dụng.'
            },
            {
                'user': 'Tôi sợ bị trả đũa nếu đi khiếu nại.',
                'detected_lang': 'Vietnamese',
                'assistant': 'Lo lắng như vậy là bình thường. Bạn vẫn có quyền khiếu nại và có thể hỏi cơ quan chức năng về cách xử lý an toàn.'
            },
            {
                'user': 'Bạn có thể tìm văn phòng lao động gần tôi không?',
                'detected_lang': 'Vietnamese',
                'assistant': 'Có thể. Bạn cho tôi biết thành phố hoặc khu vực, tôi sẽ cung cấp địa chỉ và số liên hệ phù hợp.'
            },
            {
                'user': 'Hãy tóm tắt thành checklist ngắn giúp tôi.',
                'detected_lang': 'Vietnamese',
                'assistant': 'Checklist ngắn: lưu bằng chứng, ghi timeline, chuẩn bị giấy tờ, và nộp hồ sơ tại văn phòng lao động gần nhất.'
            }
        ]
    },
    'ilo': {
        'examples': [
            {
                'user': 'Agtatrabahoak iti adu nga oras ngem awan bayad ti overtime.',
                'detected_lang': 'Ilocano',
                'assistant': 'Saan dayta a hustisya. Isuratmo dagiti oras ti trabaho, iponnok dagiti pammaneknek, ket agreklamo iti kaasidegan nga opisina ti panagtrabaho.'
            },
            {
                'user': 'Tinatengngel ti amo ko ti passport ko. Ania ti aramidek?',
                'detected_lang': 'Ilocano',
                'assistant': 'Ti passport ket personal a dokumento. Dawatenmo nga maibalik dayta, ket no saan da ibaga, agsapul ka iti tulong iti labor office wenno pulis.'
            },
            {
                'user': 'Kasanoak nga agipila iti reklamo para iti saan a nabayadan a sueldo?',
                'detected_lang': 'Ilocano',
                'assistant': 'Urayenmo dagiti ebidensya ti sueldo, agaramid ka iti nalawag a kronolohiya, ket isumitirmo ti opisyal a reklamo iti ahensya.'
            },
            {
                'user': 'Saanak ammo agfill-up iti online a porma.',
                'detected_lang': 'Ilocano',
                'assistant': 'Mabalin ka agturong iti physical office tapno matulungan ka dagiti staff iti pannaka-fill up ti porma.'
            },
            {
                'user': 'Ania dagiti dokumento nga masapul nga iyeg no agreklamo?',
                'detected_lang': 'Ilocano',
                'assistant': 'Iyegmo ti ID wenno passport, kontrata ti trabaho, payslip, listaan ti oras, ken mensahe nga pakakitaan iti amo.'
            },
            {
                'user': 'Mabutengak no agsukir ti amo no agreklamoak.',
                'detected_lang': 'Ilocano',
                'assistant': 'Natural dayta a panagbuteng. Adda karbengam nga agreklamo, ket mabalin met dagiti opisyal nga mangted iti salaknib a pamagbannog.'
            },
            {
                'user': 'Mabalin kadi nga tulongannak agsapul iti kaasidegan nga labor office?',
                'detected_lang': 'Ilocano',
                'assistant': 'Wen, mabalin. Ibagam laeng ti siudad wenno probinsyam tapno maitedko kenka ti address ken contact details.'
            },
            {
                'user': 'Kayatko ti ababa nga checklist a nalaka suruten.',
                'detected_lang': 'Ilocano',
                'assistant': 'Simple nga checklist: ipon ebidensya, isurat ti timeline, isagana dagiti dokumento, ket agturong iti labor office tapno isumitir ti reklamo.'
            }
        ]
    },
}
