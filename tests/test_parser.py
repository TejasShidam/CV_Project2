from pathlib import Path

from anpr.data.parser import extract_plate_text_from_filename, parse_annotation_xml


def test_parse_annotation_xml_extracts_number_plate_text_attributes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "car.jpg").write_bytes(b"test")

    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """<annotation>
  <filename>car.jpg</filename>
  <object>
    <name>number_plate</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>100</xmax>
      <ymax>60</ymax>
    </bndbox>
    <attributes>
      <attribute>
        <name>number_plate_text</name>
        <value>MH 12 AB 1234</value>
      </attribute>
    </attributes>
  </object>
  <object>
    <name>number_plate</name>
    <bndbox>
      <xmin>120</xmin>
      <ymin>30</ymin>
      <xmax>240</xmax>
      <ymax>80</ymax>
    </bndbox>
    <attributes>
      <attribute>
        <name>number_plate_text</name>
        <value>NUMBER</value>
      </attribute>
    </attributes>
  </object>
</annotation>""",
        encoding="utf-8",
    )

    records = parse_annotation_xml(xml_path, image_dir)
    assert len(records) == 2
    assert records[0].plate_text == "MH12AB1234"
    assert records[0].bbox == (10, 20, 100, 60)
    assert records[1].plate_text == ""
    assert records[1].object_index == 1


  def test_extract_plate_text_from_filename_supports_space_separated_plate() -> None:
    value = extract_plate_text_from_filename("KA 00 AA 5036")
    assert value == "KA00AA5036"
