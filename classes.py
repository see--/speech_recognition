def get_classes(wanted_only=False):
  if wanted_only:
    classes = 'stop down off right up go on yes left no'
    classes = classes.split(' ')
    assert len(classes) == 10
  else:
    classes =  'sheila nine stop bed four six down bird marvin cat off right seven eight up three happy go zero on wow dog yes five one tree house two left no' # noqa
    classes = classes.split(' ')
    assert len(classes) == 30
  return classes
