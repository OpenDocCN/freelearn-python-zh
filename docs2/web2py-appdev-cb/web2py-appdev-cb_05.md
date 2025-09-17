# 第五章 添加 Ajax 效果

在本章中，我们将涵盖以下食谱：

+   使用`jquery.multiselect.js`

+   创建`select_or_add`小部件

+   使用自动完成插件

+   创建下拉日期选择器

+   改进内置的`ajax`函数

+   使用滑块表示数字

+   使用 jqGrid 和 web2py

+   使用 WebGrid 改进数据表

+   Ajax 化您的搜索功能

+   创建 sparklines

# 简介

在本章中，我们讨论了 jQuery 插件与 web2py 集成的示例。这些插件有助于使表单和表格更加交互友好，从而提高应用程序的可用性。特别是，我们提供了如何通过交互式**添加选项**按钮改进多选下拉列表、如何用滑块替换输入字段以及如何使用`jqGrid`和`WebGrid`显示表格数据的示例。

# 使用`jquery.multiselect.js`

`<select multiple="true">..</select>`的默认渲染非常丑陋且不直观，尤其是在您需要选择多个非连续选项时。这并不是 HTML 的缺陷，而是大多数浏览器设计不佳。无论如何，可以使用 JavaScript 覆盖多选`select`的呈现。在这里，我们将使用一个名为`jquery.multiselect.js`的 jQuery 插件。请注意，这个 jQuery 插件作为标准插件与 PluginWiki 一起提供，但我们假设您没有使用 PluginWiki。

## 准备工作

您需要从[`abeautifulsite.net/2008/04/jquery-multiselect`](http://abeautifulsite.net/2008/04/jquery-multiselect)下载`jquery.muliselect.js`，并将相应的文件放入`static/js/jquery.multiselect.js`和`static/css/jquery.multiselect.css`。

## 如何做...

1.  在您的视图中，只需在`{{extend 'layout.html'}}:`之前添加以下内容：

    ```py
    {{
    	response.files.append('http://ajax.googleapis.com/ajax\
    		/libs/jqueryui/1.8.9/jquery-ui.js')
    	response.files.append('http://ajax.googleapis.com/ajax\
    		/libs/jqueryui/1.8.9/themes/ui-darkness/jquery-ui.css')
    	response.files.append(URL('static','js/jquery.multiSelect.js'))
     response.files.append(URL('static','css/jquery.\
    		multiSelect.css'))
    }}

    ```

1.  在`{{extend 'layout.html'}}:`之后放置以下代码：

    ```py
    <script>
    	jQuery(document).ready(function(){jQuery('[multiple]').
    		multiSelect();});
    </script>

    ```

    就这些了。你所有的多选`select`都将被优雅地样式化。

1.  考虑以下操作：

    ```py
    def index():
    	is_fruits =
    		IS_IN_SET(['Apples','Oranges','Bananas','Kiwis','Lemons'],
    		multiple=True)
    	form = SQLFORM.factory(Field('fruits','list:string',
    		requires=is_fruits))
    	if form.accepts(request,session):
    		response.flash = 'Yummy!'
    	return dict(form=form)

    ```

    可以使用以下视图尝试此操作：

    ```py
    {{
    	response.files.append('http://ajax.googleapis.com/ajax\
    		/libs/jqueryui/1.8.9/jquery-ui.js')
    	response.files.append('http://ajax.googleapis.com/ajax\
    		/libs/jqueryui/1.8.9/themes/ui-darkness/jquery-ui.css')
    	response.files.append(URL('static','js/jquery.multiSelect.js'))
    	response.files.append(URL('static','css/jquery.\
    		multiSelect.css'))
    }}
    {{extend 'layout.html}}
    <script>
    	jQuery(document).ready(function(){jQuery('[multiple]').
    		multiSelect();});
    </script>
    {{=form}}

    ```

    这是它的截图：

![如何做...](img/5467OS_05_29.jpg)

# 创建 select_or_add 小部件

此小部件将在旁边创建一个带有**添加**按钮的对象，允许用户在不访问不同屏幕的情况下即时添加新类别等。它与`IS_IN_DB`一起工作，并使用 web2py 组件和 jQueryUI 对话框。

此小部件的灵感来自可以在以下链接中找到的`OPTION_WITH_ADD_LINK`切片：

[`web2pyslices.com/main/slices/take_slice/11`](http://web2pyslices.com/main/slices/take_slice/11)

## 如何做...

1.  将以下代码放入模型文件中。例如，`models/select_or_add_widget.py:`

    ```py
    class SelectOrAdd(object):

    def __init__(self, controller=None, function=None,
    	form_title=None, button_text = None, dialog_width=450):
    		if form_title == None:
    			self.form_title = T('Add New')
    		else:
    			self.form_title = T(form_title)
    		if button_text == None:
    			self.button_text = T('Add')
    		else:
    			self.button_text = T(button_text)
    			self.dialog_width = dialog_width
    			self.controller = controller
    			self.function = function

    def widget(self, field, value):
    	#generate the standard widget for this field
    	from gluon.sqlhtml import OptionsWidget
    	select_widget = OptionsWidget.widget(field, value)

    	#get the widget's id (need to know later on so can tell
    	#receiving controller what to update)
    	my_select_id = select_widget.attributes.get('_id', None)
    	add_args = [my_select_id]

    	#create a div that will load the specified controller via ajax
    	form_loader_div = DIV(LOAD(c=self.controller, f=self.function,
    		args=add_args,ajax=True), _id=my_select_id+"_dialog-form",
    		_title=self.form_title)

    	#generate the "add" button that will appear next the options
    	#widget and open our dialog
    	activator_button = A(T(self.button_text),
    		_id=my_select_id+"_option_add_trigger")

    	#create javascript for creating and opening the dialog
    	js = 'jQuery( "#%s_dialog-form" ).dialog({autoOpen: false,
    		show: "blind", hide: "explode", width: %s});' %
    		(my_select_id, self.dialog_width)
    	js += 'jQuery( "#%s_option_add_trigger" ).click(function() {
    		jQuery( "#%s_dialog-form" ).dialog( "open" );return
    		false;}); ' % (my_select_id, my_select_id) 			#decorate
    		our activator button for good measure
    	js += 'jQuery(function() { jQuery( "#%s_option_add_trigger"
    	).button({text: true, icons: { primary: "ui-icon-circle-
    	plus"} }); });' % (my_select_id)
    	jq_script=SCRIPT(js, _type="text/javascript")

    	wrapper = DIV(_id=my_select_id+"_adder_wrapper")
    	wrapper.components.extend([select_widget, form_loader_div,
    		activator_button, jq_script])
    	return wrapper

    ```

1.  您可以使用以下方式将小部件分配给字段：

    ```py
    # Initialize the widget
    add_option = SelectOrAdd(form_title="Add a new something",
    	controller="product", function="add_category", button_text =
    	"Add New", dialog_width=500)

    ```

    此小部件接受以下参数：

    +   `form_title: string:` 这将作为 jQueryUI 对话框框的标题。默认值是`添加新内容`。

    +   `controller: string:` 这是将处理记录创建的控制器名称。

    +   `function: 字符串`。这是将处理记录创建的函数的名称。它应该创建一个表单，接受它，并准备好发出 JavaScript 与小部件交互 - 请参阅*步骤 4*中的`add_category`。

    +   `button_text: 字符串`。这是将出现在激活我们的表单对话框的按钮上的文本。默认值是`添加`。

    +   `dialog_width: 整数`。这是对话框的期望宽度（以像素为单位）。默认值为`450`。

1.  在`models/db.py`中定义您的数据库表，如下所示：

    ```py
    db.define_table('category',
    	Field('name', 'string', notnull=True, unique=True),
    	Field('description', 'text')
    )
    db.define_table('product',
    	Field('category_id', db.category, requires=IS_IN_DB(db,
    		'category.id', 'category.name')),
    	Field('name', 'string', notnull=True),
    	Field('description', 'text'),
    	Field('price', 'decimal(10,2)', notnull=True)
    )

    # assign widget to field
    	db.product.category_id.widget = add_option.widget

    ```

1.  创建您的控制器函数：

    ```py
    #This is the main function, the one your users go to
    def create():
    	#Initialize the widget
    	add_option = SelectOrAdd(form_title="Add new Product Category",
    							controller="product",
    							function="add_category",
    							button_text = "Add New")
    	#assign widget to field
    	db.product.category_id.widget = add_option.widget
    	form = SQLFORM(db.product)
    	if form.accepts(request, session):
    		response.flash = "New product created"
    	elif form.errors:
    		response.flash = "Please fix errors in form"
    	else:
    		response.flash = "Please fill in the form"

    	#you need jQuery for the widget to work; include here or just
    	#put it in your master layout.html
    	response.files.append("http://ajax.googleapis.com/ajax/\
    	libs/jqueryui/1.8.9/jquery-ui.js")
    	response.files.append("http://ajax.googleapis.com/ajax/\
    	libs/jqueryui/1.8.9/themes/smoothness/jquery-ui.css")
    	return dict(message="Create your product", form = form)

    def add_category():
    	#this is the controller function that will appear in our dialog
    	form = SQLFORM(db.category)
    	if form.accepts(request):
    		#Successfully added new item
    		#do whatever else you may want
    		#Then let the user know adding via our widget worked
    		response.flash = T("Added")
    		target = request.args[0]
    		#close the widget's dialog box
    		response.js = 'jQuery("#%s_dialog-form" ).dialog(\
    "close" );' % target

    		#update the options they can select their new category in the
    		#main form
    		response.js += \
    		"""jQuery("#%s")\
    		.append("<option value='%s'>%s</option>");""" % \
    		(target, form.vars.id, form.vars.name)
    		#and select the one they just added
    		response.js += """jQuery("#%s").val("%s");""" % \
    		(target, form.vars.id)

    		#finally, return a blank form in case for some reason they
    		#wanted to add another option
    		return form

    	elif form.errors:
    		# silly user, just send back the form and it'll still be in
    		# our dialog box complete with error messages
    		return form

    	else:
    		#hasn't been submitted yet, just give them the fresh blank
    		#form
    		return form

    ```

    这里是一个显示小部件操作的截图：

    ![如何操作...](img/5467OS_05_30.jpg)

1.  点击**添加新项**按钮，对话框就会打开。（嗯，我无法正确输入我自己的小部件名称！）。

    ![如何操作...](img/5467OS_05_31.jpg)

1.  点击**提交**，新的选项将被创建并在主表单中自动选中。

![如何操作...](img/5467OS_05_32.jpg)

您可以从以下链接在 bitbucket 上获取源代码或示例应用程序：

[`bitbucket.org/bmeredyk/web2py-select_or_add_option-widget/src`](http://https://bitbucket.org/bmeredyk/web2py-select_or_add_option-widget/src)

# 使用自动完成插件

虽然 web2py 自带自动完成插件，但其行为有点像魔法，如果不适合您，您可能更喜欢使用 jQuery 插件进行自动完成。

## 准备中

从以下网站下载必要的文件：

[`bassistance.de/jquery-plugins/jquery-plugin-autocomplete/`](http://bassistance.de/jquery-plugins/jquery-plugin-autocomplete/)

将文件解压到`static/autocomplete`。确保您有以下文件：

+   `static/autocomplete/jquery.autocomplete.js`

+   `static/autocomplete/jquery.autocomplete.css`

## 如何操作...

1.  首先，在您的模型中定义以下小部件：

    ```py
    def autocomplete_widget(field,value):
    	response.files.append(URL('static','autocomplete/jquery.\
    autocomplete.js'))
    	response.files.append(URL('static','autocomplete/jquery.\
    autocomplete.css'))
    	print response.files
    	import uuid
    	from gluon.serializers import json
    	id = "autocomplete-" + str(uuid.uuid4())
    	wrapper = DIV(_id=id)
    	inp = SQLFORM.widgets.string.widget(field,value)
    	rows = field._db(field._table['id']>0).
    		select(field,distinct=True)
    	items = [str(t[field.name]) for t in rows]
    	scr = SCRIPT("jQuery('#%s input').autocomplete({source: %s});" % \
    (id, json(items)))
    	wrapper.append(inp)
    	wrapper.append(scr)
    	return wrapper

    ```

    此小部件创建一个普通的`<input/>`小部件 inp，随后是一个注册自动完成插件的脚本。它还将一个可能值列表传递给插件，这些值是通过字段的现有值获得的。

1.  现在，在您的模型或控制器中，您只需将此小部件分配给任何字符串字段。例如：

    ```py
    db.define_table('person',Field('name'))
    db.person.name.widget = autocomplete_widget

    ```

1.  如果您想让小部件从不同的表/字段获取值，只需更改以下行：

    ```py
    rows = field._db(field._table['id']>0).select(field,distinct=True)
    items = [str(t[field.name]) for t in rows]

    ```

    将它们更改为以下内容：

    ```py
    rows = field._db(query).select(otherfield,distinct=True)
    items = [str(t[otherfield.name]) for t in rows]

    ```

### 还有更多...

这种方法的局限性在于，当小部件渲染并嵌入页面时，将获取所有可能的值。这种方法有两个局限性：

+   随着自动完成选项的增加，服务页面变得越来越慢。

+   它将您的全部数据暴露给访客

有一个解决方案。插件可以使用 Ajax 回调来获取数据。要使用 Ajax 调用远程获取项目，我们可以按以下方式修改小部件：

```py
def autocomplete_widget(field,value):
	import uuid
	id = "autocomplete-" + str(uuid.uuid4())
	callback_url = URL('get_items')
	wrapper = DIV(_id=id)
	inp = SQLFORM.widgets.string.widget(field,value)
	scr = SCRIPT("jQuery('#%s input').
		autocomplete('%s',{extraParams:{field:'%s',table:'%s'}});" % \
		(id, callback_url,field.name,field._tablename))
	wrapper.append(inp)
	wrapper.append(scr)
	return wrapper

```

现在您需要实现自己的`callback_url`。

```py
def get_items():
	MINCHARS = 2 # characters required to trigger response
	MAXITEMS = 20 # numer of items in response
	query = request.vars.q
	fieldname = request.vars.field
	tablename = request.vars.table
	if len(query.strip()) > MINCHARS and fieldname and tablename:
		field = db[tablename][fielfname]
		rows = db(field.upper().startswith(qery)).
			select(field,distinct=True,limitby=(0,MINITEMS))
		items = [str(row[fieldname]) for row in rows]
	else:
		items = []

	return '\n'.join(items)

```

这里是如何操作的示例：

![还有更多...](img/5467OS_05_33.jpg)

# 创建下拉日期选择器

有时候，你可能不喜欢正常的弹出日历选择器，而想创建一个允许分别选择年、月和日的部件，使用下拉列表。这里我们提供了一个这样的部件。

## 如何操作...

1.  在你的一个模型中编写以下部件：

    ```py
    def select_datewidget(field,value):
    	MINYEAR = 2000
    	MAXYEAR = 2020
    	import datetime
    	now = datetime.date.today()
    	dtval = value or now.isoformat()
    	year,month,day= str(dtval).split("-")
    	dt = SQLFORM.widgets.string.widget(field,value)
    	id = dt['_id']
    	dayid = id+'__day'
    	monthid = id+'__month'
    	yearid = id+'__year'
    	wrapperid = id+'__wrapper'
    	wrapper = DIV(_id=wrapperid)
    	day = SELECT([OPTION(str(i).zfill(2)) for i in range(1,32)],
    		value=day,_id=dayid)
    	month = SELECT([OPTION(datetime.date(2008,i,1).strftime('%B'),
    		_value=str(i).zfill(2)) for i in range(1,13)],
    		value=month,_id=monthid)
    	year = SELECT([OPTION(i) for i in range(MINYEAR,MAXYEAR)],
    		value=year,_id=yearid)
    	jqscr = SCRIPT("""
    		jQuery('#%s').hide();
    		var curval = jQuery('#%s').val();
    		if(curval) {
    			var pieces = curval.split('-');
    			jQuery('#%s').val(pieces[0]);
    			jQuery('#%s').val(pieces[1]);
    			jQuery('#%s').val(pieces[2]);
    		}
    		jQuery('#%s select').change(function(e) {
    			jQuery('#%s').val(
    				jQuery('#%s').val()+'-'+jQuery('#%s').val()+'-
    					'+jQuery('#%s').val());
    	});

    	""" % (id,id,yearid,monthid,dayid,
    		wrapperid,id,yearid,monthid,dayid))
    	wrapper.components.extend([month,day,year,dt,jqscr])
    	return wrapper

    ```

1.  在你的控制器中创建一个测试表单，并将字段设置为使用该部件：

    ```py
    def index():
    	form = SQLFORM.factory(
    		Field('posted','date',default=request.now,
    		widget=select_datewidget))

    	if form.accepts(request,session):
    		response.flash = "New record added"
    	return dict(form=form)

    ```

    看起来是这样的：

![如何操作...](img/5467OS_05_34.jpg)

# 改进内置的 ajax 函数

Web2py 附带一个`static/js/web2py_ajax.js`文件，该文件定义了一个 ajax 函数。它是`jQuery.ajax`的包装器，但提供了更简单的语法。然而，这个函数的设计是有意简约的。在这个菜谱中，我们向您展示如何重写它，以便在后台执行 Ajax 请求时显示旋转的图像。

## 如何操作...

1.  首先，你需要一个旋转的图标。例如，从以下网站中选择一个：[`www.freeiconsdownload.com/Free_Downloads.asp?id=585`](http://www.freeiconsdownload.com/Free_Downloads.asp?id=585)，并将其保存为`static/images/loading.gif`。

1.  然后，编辑文件`static/js/web2py_ajax.js`中的 ajax 函数，如下（对于较旧的 web2py 应用程序，此函数在`views/web2py_ajax.html`中）：`

    ```py
    function ajax(u,s,t) {
    	/* app_loading_image contains the img html
    		set in layout.html before including web2py_ajax.html */
    	jQuery("#"+t).html(app_loading_image);
    	var query="";
    	for(i=0; i<s.length; i++) {
    		if(i>0) query=query+"&";
    		query=query+encodeURIComponent(s[i])+"="+
    			encodeURIComponent(document.getElementById(s[i]).value);
    	}
    	// window.alert(loading_image);
    	jQuery.ajax({type: "POST", url: u, data: query,
    		success: function(msg) {
    			if(t==':eval') eval(msg);
    			else document.getElementById(t).innerHTML=msg;
    		}
    	});
    };

    ```

# 使用滑块表示数字

jQuery UI 附带了一个方便的滑块，可以用来表示范围中的数值字段，而不是无聊的`<input/>`标签。

## 如何操作...

1.  创建一个名为`models/plugin_slider.py`的模型文件，并定义以下内容：

    ```py
    def slider_widget(field,value):
    	response.files.append("http://ajax.googleapis.com/ajax\
    /libs/jqueryui/1.8.9/jquery-ui.js")
    	response.files.append("http://ajax.googleapis.com/ajax\
    /libs/jqueryui/1.8.9/themes/ui-darkness/jquery-ui.css")
    	id = '%s_%s' % (field._tablename,field.name)
    	wrapper = DIV(_id="slider_wrapper",_style="width: 200px;text-\
    align:center;")
    	wrapper.append(DIV(_id=id+'__slider'))
    	wrapper.append(SPAN(INPUT(_id=id, _style="display: none;"),
    		_id=id+'__value'))
    	wrapper.append(SQLFORM.widgets.string.widget(field,value))

    	wrapper.append(SCRIPT("""
    		jQuery('#%(id)s__value').text('%(value)s');
    		jQuery('#%(id)s').val('%(value)s');
    		jQuery('#%(id)s').hide();
    		jQuery('#%(id)s__slider').slider({
    			value:'%(value)s',
    			stop: function(event, ui){
    				jQuery('#%(id)s__value').text(ui.value);
    				jQuery('#%(id)s').val(ui.value);
    		}});
    		""" % dict(id=id, value=value)))
    	return wrapper

    ```

1.  创建一个测试表，并将部件设置为我们的新滑块部件：

    ```py
    db.define_table("product",
    	Field("quantity","integer", default=0))

    ```

1.  然后，通过在控制器中创建一个表单来使用滑块：

    ```py
    def index():
    	db.product.quantity.widget=slider_widget
    	form = SQLFORM(db.product)
    	if form.accepts(request,session):
    		response.flash = "Got it"
    	inventory = db(db.product).select()
    	return dict(form=form,inventory=inventory)

    ```

    ![如何操作...](img/5467OS_05_35.jpg)

# 使用 jqGrid 和 web2py

**jqGrid** 是一个基于 jQuery 构建的 Ajax 启用 JavaScript 控件，它提供了一个表示和操作表格数据的解决方案。你可以把它看作是 web2py `SQLTABLE`辅助器的替代品。jqGrid 是一个客户端解决方案，它通过 Ajax 回调动态加载数据，从而提供分页、搜索弹出、行内编辑等功能。jqGrid 已集成到 PluginWiki 中，但在这里，我们将其作为一个独立的 web2py 程序来讨论，这些程序不使用插件。jqGrid 值得有一本书来介绍，但在这里我们只讨论其基本功能和最简单的集成。

## 准备工作

你将需要 jQuery（它随 web2py 一起提供）、jQuery.UI 以及一个或多个主题，你可以直接从 Google 获取，但你还需要 jqGrid，你可以从以下地方获取：

[`www.trirand.com/blog`](http://www.trirand.com/blog)

我们还假设我们有一个包含内容的表，你可以用随机数据预先填充：

```py
from gluon.contrib.populate import populate

db.define_table('stuff',
	Field('name'),
	Field('quantity', 'integer'),
	Field('price', 'double'))

if db(db.stuff).count() == 0:
	populate(db.stuff, 50)

```

## 如何操作...

首先，你需要一个将显示 jqGrid 的辅助器，我们可以在一个模型中定义它。例如，`models/plugin_qgrid.py:`

```py
def JQGRID(table,fieldname=None, fieldvalue=None, col_widths=[],
			colnames=[], _id=None, fields=[],
			col_width=80, width=700, height=300, dbname='db'):
	# <styles> and <script> section
		response.files.append('http://ajax.googleapis.com/ajax\
/libs/jqueryui/1.8.9/jquery-ui.js')
	response.files.append('http://ajax.googleapis.com/ajax\
	/libs/jqueryui/1.8.9/themes/ui-darkness/jquery-ui.css')
	for f in ['jqgrid/ui.jqgrid.css',
				'jqgrid/i18n/grid.locale-en.js',
				'jqgrid/jquery.jqGrid.min.js']:
		response.files.append(URL('static',f))

	# end <style> and <script> section
	from gluon.serializers import json
	_id = _id or 'jqgrid_%s' % table._tablename
	if not fields:
		fields = [field.name for field in table if field.readable]
	else:
		fields = fields
	if col_widths:
		if isinstance(col_widths,(list,tuple)):
			col_widths = [str(x) for x in col_widths]
		if width=='auto':
			width=sum([int(x) for x in col_widths])
	elif not col_widths:
		col_widths = [col_width for x in fields]
		colnames = [(table[x].label or x) for x in fields]
		colmodel = [{'name':x,'index':x, 'width':col_widths[i],
					'sortable':True} \
					for i,x in enumerate(fields)]

	callback = URL('jqgrid',
					vars=dict(dbname=dbname,
								tablename=table._tablename,
								columns=','.join(fields),
								fieldname=fieldname or '',
								fieldvalue=fieldvalue,
								),
					hmac_key=auth.settings.hmac_key,
					salt=auth.user_id)
	script="""
	jQuery(function(){
	jQuery("#%(id)s").jqGrid({
	url:'%(callback)s',
	datatype: "json",
	colNames: %(colnames)s,
	colModel:%(colmodel)s,
	rowNum:10, rowList:[20,50,100],
	pager: '#%(id)s_pager',
	viewrecords: true,
	height:%(height)s
	});
	jQuery("#%(id)s").jqGrid('navGrid','#%(id)s_pager',{
	search:true,add:false,
	edit:false,del:false
	});
	jQuery("#%(id)s").setGridWidth(%(width)s,false);
	jQuery('select.ui-pg-selbox,input.ui-g-
	input').css('width','50px');
	});
	""" % dict(callback=callback, colnames=json(colnames),
				colmodel=json(colmodel),id=_id,
				height=height,width=width)

	return TAG'',
					DIV(_id=_id+"_pager"),
					SCRIPT(script))

```

我们可以这样在我们的控制中使用它：

```py
@auth.requires_login()
def index():
	return dict(mygrid = JQGRID(db.stuff))

```

这个函数简单地生成所有必需的 JavaScript，但不向它传递任何数据。相反，它传递一个回调函数 URL（`jqgrid`），该 URL 为安全起见进行了数字签名。我们需要实现这个回调。

我们可以在索引操作的同一控制器中定义回调：

```py
def jqgrid():
	from gluon.serializers import json
	import cgi
	hash_vars = 'dbname|tablename|columns|fieldname|
		fieldvalue|user'.split('|')
	if not URL.verify(request,hmac_key=auth.settings.hmac_key,
		hash_vars=hash_vars,salt=auth.user_id):
		raise HTTP(404)

	dbname = request.vars.dbname or 'db'
	tablename = request.vars.tablename or error()
	columns = (request.vars.columns or error()).split(',')
	rows=int(request.vars.rows or 25)
	page=int(request.vars.page or 0)
	sidx=request.vars.sidx or 'id'
	sord=request.vars.sord or 'asc'
	searchField=request.vars.searchField
	searchString=request.vars.searchString
	searchOper={'eq':lambda a,b: a==b,
		'nq':lambda a,b: a!=b,
		'gt':lambda a,b: a>b,
		'ge':lambda a,b: a>=b,
		'lt':lambda a,b: a<b,
		'le':lambda a,b: a<=b,
		'bw':lambda a,b: a.startswith(b),
		'bn':lambda a,b: ~a.startswith(b),
		'ew':lambda a,b: a.endswith(b),
		'en':lambda a,b: ~a.endswith(b),
		'cn':lambda a,b: a.contains(b),
		'nc':lambda a,b: ~a.contains(b),
		'in':lambda a,b: a.belongs(b.split()),
		'ni':lambda a,b: ~a.belongs(b.split())}\

	[request.vars.searchOper or 'eq']
	table=globals()[dbname][tablename]

	if request.vars.fieldname:
		names = request.vars.fieldname.split('|')
		values = request.vars.fieldvalue.split('|')
		query = reduce(lambda a,b:a&b,
			[table[names[i]]==values[i] for i in range(len(names))])

	else:
	query = table.id>0
	dbset = table._db(query)

	if searchField:
		dbset=dbset(searchOper(table[searchField],searchString))
		orderby = table[sidx]

	if sord=='desc': orderby=~orderby
		limitby=(rows*(page-1),rows*page)
		fields = [table[f] for f in columns]
		records = dbset.select(orderby=orderby,limitby=limitby,*fields)
		nrecords = dbset.count()
		items = {}
		items['page']=page
		items['total']=int((nrecords+(rows-1))/rows)
		items['records']=nrecords
		readable_fields=[f.name for f in fields if f.readable]
		def f(value,fieldname):
			r = table[fieldname].represent
		if r: value=r(value)
		try: return value.xml()
		except: return cgi.escape(str(value))
		items['rows']=[{'id':r.id,'cell':[f(r[x],x) for x in
			readable_fields]} \
			for r in records]
		return json(items)

```

`JQGRID` 辅助工具和 `jqgrid` 动作都是预制的，非常类似于 PluginWiki 的 `jgGrid` 小部件，可能不需要任何修改。`jqgrid` 动作是由辅助工具生成的代码调用的。它检查 URL 是否正确签名（用户有权访问回调）或未签名，解析请求中的所有数据以确定用户想要什么，包括从 `jqgrid` 搜索弹出窗口构建查询，并通过 JSON 在数据上执行 `select` 和 `return` 操作。

注意，您可以在多个操作中使用多个 `JQGRID(table)`，并且除了要显示的表之外，您不需要传递任何其他参数。但是，您可能希望向辅助工具传递额外的参数：

+   `fieldname` 和 `fieldvalue` 属性用于根据 `table[fieldname]==fieldvalue` 预先筛选结果

+   `col_widths` 是像素中的列宽列表

+   `colnames` 是要替换 `field.name` 的列名列表

+   `_id` 是网格的标签 ID

+   `fields` 是要显示的字段名列表

+   `col_width=80` 是每列的默认宽度

+   `width=700` 和 `height=300` 是网格的大小

+   `dbname='db'` 是回调将使用的数据库的名称，如果您有多个数据库，或者您使用的是不是 `db` 的名称

# 使用 WebGrid 提高数据表

在这个菜谱中，我们将构建一个名为 WebGrid 的模块，您可以将它视为 web2py 的 SQLTABLE 的替代品。然而，它更智能：它支持分页、排序、编辑，并且易于使用和定制。它故意设计为不需要会话或 jQuery 插件。

## 准备工作

从 [`web2pyslices.com/main/static/share/webgrid.py`](http://web2pyslices.com/main/static/share/webgrid.py) 下载 `webgrid.py`，并将其存储在 `modules/` 文件夹中。

您可以从 [`web2pyslices.com/main/static/share/web2py.app.webgrid.w2p`](http://web2pyslices.com/main/static/share/web2py.app.webgrid.w2p) 下载一个演示应用程序，但这对于 WebGrid 正常工作不是必需的。

我们将假设有一个具有 `crud` 定义的脚手架应用程序，以及以下代码：

```py
db.define_table('stuff',
	Field('name'),
	Field('location'),
	Field('quantity','integer'))

```

我们心中有一个简单的库存系统。

## 如何做到这一点...

我们将改变一下顺序。首先，我们将向您展示如何使用它。

1.  将 `webgrid.py` 模块添加到您的 `modules` 文件夹中（有关安装说明，请参阅 *准备工作* 部分）。在您的控制器中添加以下代码：

    ```py
    def index():
    	import webgrid
    	grid = webgrid.WebGrid(crud)
    	grid.datasource = db(db.stuff.id>0)
    	grid.pagesize = 10
    	return dict(grid=grid()) # notice the ()

    ```

    数据源可以是 `Set`、`Rows`、`Table` 或 `Table` 列表。也支持连接。

    ```py
    grid.datasource = db(db.stuff.id>0) 				# Set
    grid.datasource = db(db.stuff.id>0).select() 	# Rows
    grid.datasource = db.stuff 							# Table
    grid.datasource = [db.stuff,db.others] 				# list of Tables
    grid.datasource = db(db.stuff.id==db.other.thing) 	#	join

    ```

    WebGrid 的主要行组件包括 `header`、`filter`、`datarow`、`pager`、`page_total` 和 `footer`

1.  您可以使用 `action_links` 链接到 `crud` 函数。只需告诉它 `crud` 在哪里公开：

    ```py
    grid.crud_function = 'data'

    ```

1.  你可以开启或关闭`rows`：

    ```py
    grid.enabled_rows = ['header','filter',
    'pager','totals','footer','add_links']

    ```

1.  你可以控制`fields`和`field_headers`：

    ```py
    grid.fields = ['stuff.name','stuff.location','stuff.quantity']
    grid.field_headers = ['Name','Location','Quantity']

    ```

1.  你可以控制`action_links`（指向`crud`操作的链接）和`action_headers`：

    ```py
    grid.action_links = ['view','edit','delete']
    grid.action_headers = ['view','edit','delete']

    ```

1.  你可能需要修改`crud.settings.[action]_next`，以便在完成操作后重定向到你的 WebGrid 页面：

    ```py
    if request.controller == 'default' and request.function == 'data':
    	if request.args:
    		crud.settings[request.args(0)+'_next'] = URL('index')

    ```

1.  你可以为数值字段获取页面`总计`：

    ```py
    grid.totals = ['stuff.quantity']

    ```

1.  你可以在列上设置`过滤器`：

    ```py
    grid.filters = ['stuff.name','stuff.created']

    ```

1.  你可以修改`过滤器`使用的`查询`（如果你的数据源是`Rows`对象，则不可用；使用`rows.find`）：

    ```py
    grid.filter_query = lambda f,v: f==v

    ```

1.  你可以控制哪些请求`变量`可以覆盖`grid`设置：

    ```py
    grid.allowed_vars =
    	['pagesize','pagenum','sortby','ascending','groupby','totals']

    ```

    当渲染单元格时，WebGrid 将使用字段表示函数，如果存在。如果你需要更多的控制，你可以完全覆盖渲染行的方式。

1.  渲染每一行的函数可以被替换成你自己的`lambda`或函数：

    ```py
    grid.view_link = lambda row: ...
    grid.edit_link = lambda row: ...
    grid.delete_link = lambda row: ...
    grid.header = lambda fields: ...
    grid.datarow = lambda row: ...
    grid.footer = lambda fields: ...
    grid.pager = lambda pagecount: ...
    grid.page_total = lambda:

    ```

1.  这里有一些有用的变量，用于构建你自己的行：

    ```py
    grid.joined # tells you if your datasource is a join
    grid.css_prefix # used for css
    grid.tablenames
    grid.response # the datasource result
    grid.colnames # column names of datasource result
    grid.pagenum
    grid.pagecount
    grid.total # the count of datasource result

    ```

    例如，让我们自定义页脚：

    ```py
    grid.footer = lambda fields : TFOOT(TD("This is my footer" ,
    	_colspan=len(grid.action_links)+len(fields),
    	_style="text-align:center;"),
    	_class=grid.css_prefix + '-webgrid footer')

    ```

1.  你还可以自定义消息：

    ```py
    grid.messages.confirm_delete = 'Are you sure?'
    grid.messages.no_records = 'No records'
    grid.messages.add_link = '[add %s]'
    grid.messages.page_total = "Total:"

    ```

1.  你还可以使用`row_created`事件在行创建时修改该行。让我们在表头中添加一个列：

    ```py
    def on_row_created(row,rowtype,record):
    	if rowtype=='header':
    		row.components.append(TH(' '))
    grid.row_created = on_row_created

    ```

1.  让我们将操作链接移到右侧：

    ```py
    def links_right(tablerow,rowtype,rowdata):
    	if rowtype != 'pager':
    	links = tablerow.components[:3]
    	del tablerow.components[:3]
    	tablerow.components.extend(links)
    grid.row_created = links_right

    ```

    ![如何操作...](img/5467OS_05_36.jpg)

如果你在同一页面上使用多个网格，它们必须具有唯一的名称。

# Ajax 化你的搜索功能

在这个菜谱中，我们描述了视频中展示的代码：

[`www.youtube.com/watch?v=jGuW43sdv6E`](http://www.youtube.com/watch?v=jGuW43sdv6E)

它与自动完成非常相似。它允许你在输入字段中输入代码，通过 Ajax 将文本发送到服务器，并显示服务器返回的结果。它可以用于执行实时搜索。它与自动完成不同，因为文本不一定来自一个表（它可以来自服务器端实现的更复杂的搜索条件），并且结果不用于填充输入字段。

## 如何操作...

1.  我们需要从一个模型开始，在这个例子中，我们选择了这个模型：

    ```py
    db.define_table('country',
    	Field('iso'),
    	Field('name'),
    	Field('printable_name'),
    	Field('iso3'),
    	Field('numcode'))

    ```

1.  我们用以下数据填充此模型：

    ```py
    if not db(db.country).count():
    	for (iso,name,printable_name,iso3,numcode) in [
    		('UY','URUGUAY','Uruguay','URY','858'),
    		('UZ','UZBEKISTAN','Uzbekistan','UZB','860'),
    		('VU','VANUATU','Vanuatu','VUT','548'),
    		('VE','VENEZUELA','Venezuela','VEN','862'),
    		('VN','VIETNAM','Viet Nam','VNM','704'),
    		('VG','VIRGIN ISLANDS, BRITISH','Virgin Islands,
    			British','VGB','092'),
    		('VI','VIRGIN ISLANDS, U.S.','Virgin Islands,
    			U.s.','VIR','850'),
    		('EH','WESTERN SAHARA','Western Sahara','ESH','732'),
    		('YE','YEMEN','Yemen','YEM','887'),
    		('ZM','ZAMBIA','Zambia','ZMB','894'),
    		('ZW','ZIMBABWE','Zimbabwe','ZWE','716')]:
    db.country.insert(iso=iso,name=name,printable_name=printable_name,
    	iso3=iso3,numcode=numcode)

    ```

1.  创建以下 CSS 文件`static/css/livesearch.css`：

    ```py
    #livesearchresults {
    	background: #ffffff;
    	padding: 5px 10px;
    	max-height: 400px;
    	overflow: auto;
    	position: absolute;
    	z-index: 99;
    	border: 1px solid #A9A9A9;
    	border-width: 0 1px 1px 1px;
    	-webkit-box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.3);
    	-moz-box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.3);
    	-box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.3);
    }

    #livesearchresults a{
    	color:#666666;
    }
    input#livesearch {
    	font-size:12px;
    	color:#666666;
    	background-color:#ffffff;
    	padding-top:5px;
    	width:200px;
    	height:20px;
    	border:1px solid #999999;
    }

    ```

1.  创建以下 JavaScript 文件`static/js/livesearch.js`：

    ```py
    function livesearch(value){
    	if(value != ""){
    		jQuery("#livesearchresults").show();
    		jQuery.post(livesearch_url,
    			{keywords:value},
    			function(result){
    				jQuery("#livesearchresults").html(result);
    			}
    		);
    	}

    	else{
    		jQuery("#livesearchresults").hide();
    	}
    }

    function updatelivesearch(value){
    	jQuery("#livesearch").val(value);jQuery("#livesearchresults").
    		hide();
    }

    jQuery(function(){jQuery("#livesearchresults").hide();});

    ```

1.  现在创建一个简单的控制器动作：

    ```py
    def index():
    	return dict()

    ```

1.  简单控制器动作关联到以下`views/default/index.html`，它使用了在*步骤 3*和*步骤 4*中创建的 livesearch JS 和 CSS：

    ```py
    <script type="text/javascript">
    	/* url definition for livesearch ajax call */
    	var livesearch_url = "{{=URL('ajaxlivesearch')}}";
    </script>
    {{response.files.append(URL('static','css/livesearch.css'))}}
    {{response.files.append(URL('static','js/livesearch.js'))}}
    {{extend 'layout.html'}}

    <label for="livesearch">Search country:</label><br />
    <input type="text" id="livesearch" name="country" autocomplete="off" onkeyup="livesearch(this.value);" /><br />
    <div id="livesearchresults"></div>

    ```

1.  最后，在`index`函数相同的控制器中，实现 Ajax 回调：

    ```py
    def ajaxlivesearch():
    	keywords = request.vars.keywords
    	print "Keywords: " + str(keywords)

    	if keywords:
    		query = reduce(lambda a,b:a&b,
    			[db.country.printable_name.contains(k) for k in \
    			keywords.split()])

    	countries = db(query).select()
    	items = []

    	for c in countries:
    		items.append(DIV(A(c.printable_name, _href="#",
    			_id="res%s"%c.iso,
    			_onclick="updatelivesearch(jQuery('#res%s').
    			html())"%c.iso)))
    	return DIV(*items)

    ```

    这就是它的样子：

![如何操作...](img/5467OS_05_37.jpg)

# 创建 sparklines

`Sparklines`是小型图表，通常嵌入在文本中，用于总结时间序列或类似信息。`jquery.sparklines`插件提供了几种不同的图表样式和有用的显示选项。你可以将 sparklines 插件与`jquery.timers`插件结合使用，以显示实时变化的数据。这个菜谱展示了实现这一点的其中一种方法。

Sparkline 图表在需要直观比较大量相似数据系列的应用程序中非常有用。以下是一个链接，指向*爱德华·图费尔的《美丽证据》*一书中关于更多信息的一个章节：

[`www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0001OR`](http://www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0001OR)

我们将创建一个索引，显示 5 到 25 个条形图，展示随机数字，反向排序以模拟帕累托图。图表每秒更新一次，以从服务器获取的新数据。

显示效果如下：

![创建 sparklines](img/5467OS_05_38.jpg)

此示例假设您可以使用单个 JSON 查询一次性获取所有 sparklines 的数据，并且您在视图渲染时知道要显示多少个图表。技巧是选择一个合适的方案来生成图形 ID，在这种情况下是`["dynbar0", "dynbar1",....]`，并且使用从 JSON 服务函数返回的相同 ID 字符串作为字典的键。这使得使用 web2py 视图模板方法生成`jquery.sparkline()`调用以更新从服务函数返回的 sparklines 变得简单。

## 如何做到这一点...

1.  首先，您需要下载以下内容：

    +   [`plugins.jquery.com/project/sparklines, 进入 "static/js/jquery.sparkline.js"`](http://plugins.jquery.com/project/sparklines)

    +   以及计时器，[`plugins.jquery.com/project/timers`](http://plugins.jquery.com/project/timers)，进入`static/js/jquery.timers-1.2.js`

1.  然后，在您的`layout.html`中，在包含`web2py_ajax.html`之前，添加以下内容：

    ```py
    response.files.append(URL('static','js/jquery.sparkline.js'))
    response.files.append(URL('static','js/jquery.timers-1.2.js'))

    ```

1.  将以下操作添加到您的控制器中：

    ```py
    def index():
    	return dict(message="hello from sparkline.py",
    		ngraphs=20, chartmin=0, chartmax=20)

    def call():
    	return service()

    @service.json
    def sparkdata(ngraphs,chartmin,chartmax):
    	import random
    	ngraphs = int(ngraphs)
    	chartmin = int(chartmin)
    	chartmax = int(chartmax)

    	d = dict()
    	for n in xrange(ngraphs):
    	id = "dynbar" + str(n)
    	### data for bar graph.
    	### 9 random ints between chartmax and chartmin
    	data = [random.choice(range(chartmin,chartmax))\
    			for i in xrange(9)]
    	### simulate a Pareto plot
    	data.sort()
    	data.reverse()
    	d[id] = data
    return d

    ```

1.  然后，创建`views/default/index.html`，如下所示：

    ```py
    {{extend 'layout.html'}}
    {{
    	chartoptions =
    		XML("{type:'bar',barColor:'green','chartRangeMin':'%d',
    		'chartRangeMax':'%d'}" % (chartmin,chartmax))
    		jsonurl = URL('call/json/sparkdata/\
    		%(ngraphs)d/%(chartmin)d/%(chartmax)d' % locals())
    }}

    <script type="text/javascript">
    	jQuery(function() {
    		jQuery(this).everyTime(1000,function(i) {
    			jQuery.getJSON('{{=jsonurl}}', function(data) {
    				{{for n in xrange(ngraphs):}}
    				jQuery("#dynbar{{=n}}").sparkline(data.dynbar{{=n}},
    				{{ =chartoptions }} );
    				{{pass}}
    				});
    		});
    	});
    </script>
    <h1>This is the sparkline.html template</h1>
    {{for n in xrange(ngraphs):}}
    <p>
    	Bar chart with dynamic data: <span id="dynbar{{=n}}"
    		class="dynamicbar">Loading..</span>
    </p>
    {{pass}}
    {{=BEAUTIFY(response._vars)}}

    ```
